// Export all services from a central location
export * from "./api"
export * from "./orderService"
export * from "./productService"
export * from "./authService"

// Re-export commonly used types
export type {
  Order,
  OrderItem
} from "./orderService"

export type {
  Product,
  CreateProductData,
  UpdateProductData
} from "./productService"

export type {
  User,
  LoginCredentials,
  RegisterData,
  AuthResponse
} from "./authService"

export type {
  ApiResponse,
  ApiError
} from "./api"

// Re-export commonly used functions for convenience
export {
  fetchOrders,
  fetchProducts,
  updateOrderStatus,
  updatePaymentStatus,
  updateOrder,
  addProductToOrder,
  removeOrderItem,
  copyToClipboard
} from "./orderService"
