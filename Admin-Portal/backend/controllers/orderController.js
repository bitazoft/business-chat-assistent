import orderService from '../services/orderService.js';


const getAllOrders = async (req, res) => {
    const { seller_id } = req.params;
    try {
      const orders = await orderService.getOrdersBySellerId(seller_id);
      console.log("Fetched orders:", orders);
      res.json({ orders });
    } catch (error) {
      console.error("Error fetching orders:", error);
      res.status(500).json({ error: "Failed to fetch orders" });
    }
  }

const getOrderById = async (req, res) => {
    const { id } = req.params;
    try {
      const order = await orderService.getOrderById(id);
      console.log("Fetched order:", order);
      if (!order) {
        return res.status(404).json({ error: "Order not found" });
      }
      res.json({ order });
    } catch (error) {
      console.error("Error fetching order:", error);
      res.status(500).json({ error: "Failed to fetch order" });
    }
  }

const createOrder = async (req, res) => {
    const orderData = req.body;
    try {
      const newOrder = await orderService.createOrder(orderData);
      res.status(201).json({ order: newOrder });
    } catch (error) {
      console.error("Error creating order:", error);
      res.status(500).json({ error: "Failed to create order" });
    }
  }

const updateOrder = async (req, res) => {
    const { id } = req.params;
    const orderData = req.body;
    try {
      const updatedOrder = await orderService.updateOrder(id, orderData);
      console.log("Updated order:", updatedOrder);
      res.json({ order: updatedOrder });
    } catch (error) {
      console.error("Error updating order:", error);
      res.status(500).json({ error: "Failed to update order" });
    }
  }

const deleteOrder = async (req, res) => {
    const { id } = req.params;
    try {
      await orderService.deleteOrder(id);
      console.log("Deleted order:", id);
      res.status(204).end();
    } catch (error) {
      console.error("Error deleting order:", error);
      res.status(500).json({ error: "Failed to delete order" });
    }
  };
const updateOrderStatus = async (req, res) => {
    const { id } = req.params;
    const { status } = req.body;
    try {
      const updatedOrder = await orderService.updateOrderStatus(id, status);
      console.log("Updated order status:", updatedOrder);
      res.json({ order: updatedOrder });
    } catch (error) {
      console.error("Error updating order status:", error);
      res.status(500).json({ error: "Failed to update order status" });
    }
  };

const addOrderItem = async (req, res) => {
  const { id } = req.params; // order id
  const { productId, quantity, price } = req.body;
  
  try {
    const updatedOrder = await orderService.addOrderItem(id, { productId, quantity, price });
    console.log("Added order item:", updatedOrder);
    res.json({ order: updatedOrder });
  } catch (error) {
    console.error("Error adding order item:", error);
    res.status(500).json({ error: error.message || "Failed to add order item" });
  }
};

const removeOrderItem = async (req, res) => {
  const { id, itemId } = req.params; // order id and item id
  
  try {
    const updatedOrder = await orderService.removeOrderItem(id, itemId);
    console.log("Removed order item:", updatedOrder);
    res.json({ order: updatedOrder });
  } catch (error) {
    console.error("Error removing order item:", error);
    res.status(500).json({ error: error.message || "Failed to remove order item" });
  }
};

export {
  getAllOrders,
  getOrderById,
  createOrder,
  updateOrder,
  deleteOrder,
  updateOrderStatus,
  addOrderItem,
  removeOrderItem
};
