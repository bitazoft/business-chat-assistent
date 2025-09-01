// Export all services from a central location
export * from "./api"
export * from "./orderService"
export * from "./productService"
export * from "./authService"
export * from "./businessService"

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
  BusinessDetails,
  UpdateBusinessDetailsData,
  UpdatePasswordData
} from "./businessService"

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
