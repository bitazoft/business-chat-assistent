import { toast } from "sonner"
import { api, handleApiError, handleApiSuccess, cacheManager, cachedRequest } from "./api"

// Types for the service
export interface OrderItem {
  id: string
  name: string
  productId: string
  quantity: number
  price: number
  total: number
}

export interface Product {
  id: string
  name: string
  price: number
  stock: number
  description?: string
}

export interface Order {
  id: string
  customer: string
  phone: string
  address: string
  email: string
  netValue: number
  date: string
  status: string
  shippingCost: number
  orderItems: OrderItem[]
  notes?: string
  paymentStatus?: string
  paymentMethod?: string
  paymentProofUrl?: string
}

// Transform backend order data to frontend format
const transformOrderData = (order: any): Order => {
  const transformedOrderItems = (order.order_items || []).map((item: any) => ({
    id: item.id?.toString() || '',
    name: item.products?.name || item.product_name || item.name || 'Unknown Product',
    productId: item.product_id?.toString() || '',
    quantity: item.quantity || 0,
    price: item.price || 0,
    total: item.total || (item.quantity * item.price) || 0
  }))
  
  return {
    id: order.id?.toString() || '',
    customer: order.customers.name || 'Unknown Customer',
    phone: order.customers.number1 || '',
    address: order.customers.address || '',
    netValue: order.total_amount || 0,
    shippingCost: order.shipping_cost || 0,
    email: order.customers.email || '',
    date: order.created_at ? new Date(order.created_at).toISOString().split('T')[0] : '',
    status: order.status === 'pending' ? 'Pending' : 
           order.status === 'completed' ? 'Completed' :
           order.status === 'processing' ? 'Processing' :
           order.status === 'shipped' ? 'Shipped' :
           order.status === 'cancelled' ? 'Cancelled' :
           'Pending', // default to Pending
    orderItems: transformedOrderItems,
    notes: order.notes || '',
    paymentStatus: order.payment_status || 'Pending',
    paymentMethod: order.payment_method || 'Unknown',
    paymentProofUrl: order.payment_proof || ''
  }
}

// Transform backend product data to frontend format
const transformProductData = (product: any): Product => ({
  id: product.id?.toString() || '',
  name: product.name || 'Unknown Product',
  price: parseFloat(product.price) || 0,
  stock: product.stock || 0,
  description: product.description || ''
})

/**
 * Fetch all orders for a specific seller
 */
export const fetchOrders = async (sellerId: string): Promise<Order[]> => {
  if (!sellerId) {
    const error = "Seller ID is required"
    handleApiError(new Error(error))
    throw new Error(error)
  }

  try {
    const cacheKey = `orders-${sellerId}`
    
    const data = await cachedRequest(
      cacheKey,
      () => api.get(`/orders/${sellerId}`),
      300000 // 5 minutes cache
    )

    console.log('Raw backend data:', data)

    const orders: Order[] = (data.orders || []).map(transformOrderData)
    console.log('Transformed orders:', orders)

    return orders
  } catch (error) {
    handleApiError(error, "Failed to fetch orders")
    throw error
  }
}

/**
 * Fetch all products for a specific seller
 */
export const fetchProducts = async (sellerId: string): Promise<Product[]> => {
  if (!sellerId) {
    const error = "Seller ID is required"
    handleApiError(new Error(error))
    throw new Error(error)
  }

  try {
    const cacheKey = `products-${sellerId}`
    
    const data = await cachedRequest(
      cacheKey,
      () => api.get(`/products/getAll/${sellerId}`),
      300000 // 5 minutes cache
    )

    console.log('Raw products data:', data)
    
    const transformedProducts = (data.products || []).map(transformProductData)
    console.log('Transformed products:', transformedProducts)
    
    return transformedProducts
  } catch (error) {
    handleApiError(error, "Failed to fetch products")
    throw error
  }
}

/**
 * Update order status
 */
export const updateOrderStatus = async (orderId: string, newStatus: string): Promise<void> => {
  try {
    await api.put(`/orders/${orderId}`, { status: newStatus.toLowerCase() })
    
    // Clear related cache
    cacheManager.clear()
    
    handleApiSuccess(`Order status updated to ${newStatus}`)
  } catch (error) {
    handleApiError(error, "Failed to update order status")
    throw error
  }
}

/**
 * Update payment status
 */
export const updatePaymentStatus = async (orderId: string, newPaymentStatus: string): Promise<void> => {
  try {
    await api.put(`/orders/${orderId}`, { payment_status: newPaymentStatus.toLowerCase() })
    
    // Clear related cache
    cacheManager.clear()
    
    handleApiSuccess(`Payment status updated to ${newPaymentStatus}`)
  } catch (error) {
    handleApiError(error, "Failed to update payment status")
    throw error
  }
}

/**
 * Update entire order
 */
export const updateOrder = async (order: Order): Promise<void> => {
  try {
    const updatedOrder = {
      customer: order.customer,
      phone: order.phone,
      address: order.address,
      email: order.email,
      net_value: order.netValue,
      shipping_cost: order.shippingCost,
      status: order.status.toLowerCase(),
      notes: order.notes || '',
      payment_status: order.paymentStatus || 'pending',
      payment_method: order.paymentMethod || 'Unknown',
      payment_proof: order.paymentProofUrl || '',
      order_items: order.orderItems.map(item => ({
        id: item.id,
        quantity: item.quantity,
        product_id: item.productId,
        price: item.price,
        total: item.total
      }))
    }

    await api.put(`/orders/${order.id}`, updatedOrder)
    
    // Clear related cache
    cacheManager.clear()
    
    handleApiSuccess("Order updated successfully")
  } catch (error) {
    handleApiError(error, "Failed to update order")
    throw error
  }
}

/**
 * Delete an order
 */
export const deleteOrder = async (orderId: string): Promise<void> => {
  try {
    await api.delete(`/orders/${orderId}`)
    
    // Clear related cache
    cacheManager.clear()
    
    handleApiSuccess("Order deleted successfully")
  } catch (error) {
    handleApiError(error, "Failed to delete order")
    throw error
  }
}

/**
 * Create a new order
 */
export const createOrder = async (orderData: Omit<Order, 'id' | 'date'>): Promise<Order> => {
  try {
    const newOrder = {
      customer: orderData.customer,
      phone: orderData.phone,
      address: orderData.address,
      email: orderData.email,
      net_value: orderData.netValue,
      shipping_cost: orderData.shippingCost,
      status: orderData.status.toLowerCase(),
      notes: orderData.notes || '',
      payment_status: orderData.paymentStatus || 'pending',
      payment_method: orderData.paymentMethod || 'Unknown',
      payment_proof: orderData.paymentProofUrl || '',
      order_items: orderData.orderItems.map(item => ({
        quantity: item.quantity,
        product_id: item.productId,
        price: item.price,
        total: item.total
      }))
    }

    const data = await api.post('/orders', newOrder)
    
    // Clear related cache
    cacheManager.clear()
    
    handleApiSuccess("Order created successfully")
    
    // Transform and return the created order
    return {
      id: data.id?.toString() || '',
      customer: data.customer || '',
      phone: data.phone || '',
      address: data.address || '',
      email: data.email || '',
      netValue: data.net_value || 0,
      shippingCost: data.shipping_cost || 0,
      date: data.created_at ? new Date(data.created_at).toISOString().split('T')[0] : '',
      status: data.status || 'Pending',
      orderItems: data.order_items || [],
      notes: data.notes || '',
      paymentStatus: data.payment_status || 'Pending',
      paymentMethod: data.payment_method || 'Unknown',
      paymentProofUrl: data.payment_proof || ''
    }
  } catch (error) {
    handleApiError(error, "Failed to create order")
    throw error
  }
}

/**
 * Add a product to an existing order
 */
export const addProductToOrder = async (orderId: string, productId: string, quantity: number, price: number): Promise<any> => {
  try {
    const data = await api.post(`/orders/${orderId}/items`, {
      productId,
      quantity,
      price
    })
    
    // Clear related cache
    cacheManager.clear()
    
    handleApiSuccess("Product added to order successfully")
    return data
  } catch (error) {
    handleApiError(error, "Failed to add product to order")
    throw error
  }
}

/**
 * Remove an item from an order
 */
export const removeOrderItem = async (orderId: string, itemId: string): Promise<any> => {
  try {
    const data = await api.delete(`/orders/${orderId}/items/${itemId}`)
    
    // Clear related cache
    cacheManager.clear()
    
    handleApiSuccess("Item removed from order")
    return data
  } catch (error) {
    handleApiError(error, "Failed to remove order item")
    throw error
  }
}

/**
 * Get order details with items
 */
export const getOrderDetails = async (orderId: string): Promise<Order> => {
  try {
    const cacheKey = `order-details-${orderId}`
    
    const data = await cachedRequest(
      cacheKey,
      () => api.get(`/orders/${orderId}`),
      300000 // 5 minutes cache
    )

    return transformOrderData(data.order || data)
  } catch (error) {
    handleApiError(error, "Failed to fetch order details")
    throw error
  }
}

// Utility function to copy text to clipboard
export const copyToClipboard = async (text: string, type: string): Promise<void> => {
  try {
    await navigator.clipboard.writeText(text)
    toast.success(`${type} copied to clipboard`)
  } catch (error) {
    console.error("Failed to copy to clipboard:", error)
    toast.error("Failed to copy to clipboard")
    throw error
  }
}
