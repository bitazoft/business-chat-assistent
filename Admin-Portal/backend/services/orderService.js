import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

// Helper function to calculate total from order items
function calculateOrderTotal(orderItems) {
  return orderItems.reduce((total, item) => {
    return total + (parseFloat(item.price) * parseInt(item.quantity));
  }, 0);
}

async function getOrdersBySellerId(seller_id) {
  try {
    return await prisma.orders.findMany({
      where: { seller_id: parseInt(seller_id) },
      include: { 
        order_items: {
          include: {
            products: true
          }
        },
        customers: true
      },
    });
  } catch (error) {
    console.error("Error fetching orders:", error);
    throw new Error("Failed to fetch orders");
  }
}

async function getOrderById(id) {
  try {
    return await prisma.orders.findUnique({
      where: { id: parseInt(id) },
      include: { 
        order_items: {
          include: {
            products: true
          }
        },
        customers: true
      },
    });
  } catch (error) {
    console.error("Error fetching order:", error);
    throw new Error("Failed to fetch order");
  }
}

async function createOrder(data) {
  try {
    return await prisma.orders.create({
      data,
    });
  } catch (error) {
    console.error("Error creating order:", error);
    throw new Error("Failed to create order");
  }
}

async function updateOrder(id, data) {
  try {
    // Validate input
    if (!id) {
      throw new Error("Order ID is required");
    }

    const {
      customer_name,
      number1,
      address,
      status,
      email,
      shipping_cost,
      total_amount,
      notes,
      order_items,
      ...orderData
    } = data;

    // Use transaction to ensure data consistency
    const result = await prisma.$transaction(async (tx) => {
      // Get the current order to find customer_id
      const currentOrder = await tx.orders.findUnique({
        where: { id: parseInt(id) },
        include: { customers: true }
      });

      if (!currentOrder) {
        throw new Error("Order not found");
      }

      // Update customer information if provided
      if (customer_name || number1 || address) {
        const customerUpdateData = {};
        if (customer_name) customerUpdateData.name = customer_name;
        if (number1) customerUpdateData.number1 = number1;
        if (address) customerUpdateData.address = address;
        if (email) customerUpdateData.email = email;

        if (Object.keys(customerUpdateData).length > 0) {
          await tx.customers.update({
            where: { id: currentOrder.customer_id },
            data: customerUpdateData
          });
        }
      }

      // Handle order items update
      if (order_items && Array.isArray(order_items) && order_items.length > 0) {
        // Update existing order items by their IDs
        for (const item of order_items) {
          if (!item.id || !item.quantity || !item.price) {
            throw new Error("Each order item must have id, quantity, and price");
          }

          // Get current order item to calculate stock difference
          console.log('Processing order item:', item);
          if(item.id==="-1"){
            // Create new order item
            console.log('Creating new order item with product_id:', item.product_id);
            if (!item.product_id) {
              throw new Error("Product ID is required for new order items");
            }
            
            // Check if product exists and has enough stock
            const product = await tx.products.findUnique({
              where: { id: parseInt(item.product_id) }
            });
            
            if (!product) {
              throw new Error(`Product with ID ${item.product_id} not found`);
            }
            
            const requestedQuantity = parseInt(item.quantity);
            if (product.stock < requestedQuantity) {
              throw new Error(`Insufficient stock for product ${product.name}. Available: ${product.stock}, Requested: ${requestedQuantity}`);
            }
            
            const newOrderItem = await tx.order_items.create({
              data: {
                order_id: parseInt(id),
                product_id: parseInt(item.product_id),
                quantity: requestedQuantity,
                price: parseFloat(item.price)
              }
            });
            
            // Update product stock
            await tx.products.update({
              where: { id: parseInt(item.product_id) },
              data: { stock: product.stock - requestedQuantity }
            });
            
            console.log('Created new order item:', newOrderItem);
          }else{
            // Update existing order item
            const currentOrderItem = await tx.order_items.findUnique({
              where: { id: parseInt(item.id) },
              include: { products: true }
            });
            if (!currentOrderItem) {
            throw new Error(`Order item with id ${item.id} not found`);
            }

            const newQuantity = parseInt(item.quantity);
          const oldQuantity = currentOrderItem.quantity;
          const quantityDifference = newQuantity - oldQuantity;

          // Update order item
          await tx.order_items.update({
            where: { 
              id: parseInt(item.id) // item.id is the order_item id
            },
            data: {
              quantity: newQuantity,
              price: parseFloat(item.price)
            }
          });

          // Update product stock if quantity changed
          if (quantityDifference !== 0 && currentOrderItem.product_id) {
            const currentProduct = await tx.products.findUnique({
              where: { id: currentOrderItem.product_id }
            });

            if (currentProduct) {
              const newStock = currentProduct.stock - quantityDifference;
              
              // Ensure stock doesn't go negative
              if (newStock < 0) {
                throw new Error(`Insufficient stock for product ${currentProduct.name}. Available: ${currentProduct.stock}, Requested increase: ${quantityDifference}`);
              }

              await tx.products.update({
                where: { id: currentOrderItem.product_id },
                data: { stock: newStock }
              });
            }
          }
        }
          }

      }

      // Prepare order update data (only fields that exist in the orders table)
      const orderUpdateData = {};
      if (status) orderUpdateData.status = status;
      if (shipping_cost !== undefined) orderUpdateData.shipping_cost = shipping_cost.toString();
      if (notes !== undefined) orderUpdateData.notes = notes;
      
      // Calculate total amount from order items if not provided
      if (order_items && order_items.length > 0) {
        orderUpdateData.total_amount = calculateOrderTotal(order_items);
      } else if (total_amount !== undefined) {
        orderUpdateData.total_amount = parseFloat(total_amount);
      }
      
      // Add any other valid order fields from orderData
      Object.keys(orderData).forEach(key => {
        if (['payment_method', 'payment_status', 'payment_proof'].includes(key)) {
          orderUpdateData[key] = orderData[key];
        }
      });

      // Update the order
      const updatedOrder = await tx.orders.update({
        where: { id: parseInt(id) },
        data: orderUpdateData,
        include: {
          order_items: {
            include: {
              products: true
            }
          },
          customers: true
        }
      });

      return updatedOrder;
    });

    return result;
  } catch (error) {
    console.error("Error updating order:", error);
    throw new Error(`Failed to update order: ${error.message}`);
  }
}

async function deleteOrder(id) {
  try {
    return await prisma.orders.delete({
      where: { id },
    });
  } catch (error) {
    console.error("Error deleting order:", error);
    throw new Error("Failed to delete order");
  }
}

async function updateOrderStatus(id, status) {
  try {
    return await prisma.orders.update({
      where: { id: parseInt(id) },
      data: { status: status.toLowerCase() },
      include: {
        order_items: {
          include: {
            products: true
          }
        },
        customers: true
      }
    });
  } catch (error) {
    console.error("Error updating order status:", error);
    throw new Error("Failed to update order status");
  }
}

async function addOrderItem(orderId, itemData) {
  try {
    const { productId, quantity, price } = itemData;
    
    // Validate input
    if (!orderId || !productId || !quantity || !price) {
      throw new Error("Order ID, product ID, quantity, and price are required");
    }

    return await prisma.$transaction(async (tx) => {
      // Check if order exists
      const order = await tx.orders.findUnique({
        where: { id: parseInt(orderId) }
      });

      if (!order) {
        throw new Error("Order not found");
      }

      // Check if product exists and has sufficient stock
      const product = await tx.products.findUnique({
        where: { id: parseInt(productId) }
      });

      if (!product) {
        throw new Error("Product not found");
      }

      if (product.stock < parseInt(quantity)) {
        throw new Error(`Insufficient stock for product ${product.name}. Available: ${product.stock}, Requested: ${quantity}`);
      }

      // Create the order item
      await tx.order_items.create({
        data: {
          order_id: parseInt(orderId),
          product_id: parseInt(productId),
          quantity: parseInt(quantity),
          price: parseFloat(price)
        }
      });

      // Update product stock
      await tx.products.update({
        where: { id: parseInt(productId) },
        data: { stock: product.stock - parseInt(quantity) }
      });

      // Return updated order with items
      return await tx.orders.findUnique({
        where: { id: parseInt(orderId) },
        include: {
          order_items: {
            include: {
              products: true
            }
          },
          customers: true
        }
      });
    });
  } catch (error) {
    console.error("Error adding order item:", error);
    throw new Error(`Failed to add order item: ${error.message}`);
  }
}

async function removeOrderItem(orderId, itemId) {
  try {
    if (!orderId || !itemId) {
      throw new Error("Order ID and item ID are required");
    }

    return await prisma.$transaction(async (tx) => {
      // Check if order exists
      const order = await tx.orders.findUnique({
        where: { id: parseInt(orderId) }
      });

      if (!order) {
        throw new Error("Order not found");
      }

      // Get the order item to restore stock
      const orderItem = await tx.order_items.findUnique({
        where: { id: parseInt(itemId) },
        include: { products: true }
      });

      if (!orderItem) {
        throw new Error("Order item not found");
      }

      if (orderItem.order_id !== parseInt(orderId)) {
        throw new Error("Order item does not belong to this order");
      }

      // Restore product stock
      if (orderItem.product_id) {
        await tx.products.update({
          where: { id: orderItem.product_id },
          data: { 
            stock: {
              increment: orderItem.quantity
            }
          }
        });
      }

      // Delete the order item
      await tx.order_items.delete({
        where: { id: parseInt(itemId) }
      });

      // Return updated order with items
      return await tx.orders.findUnique({
        where: { id: parseInt(orderId) },
        include: {
          order_items: {
            include: {
              products: true
            }
          },
          customers: true
        }
      });
    });
  } catch (error) {
    console.error("Error removing order item:", error);
    throw new Error(`Failed to remove order item: ${error.message}`);
  }
}

export default{
  getOrdersBySellerId,
  getOrderById,
  createOrder,
  updateOrder,
  updateOrderStatus,
  deleteOrder,
  addOrderItem,
  removeOrderItem,
};
