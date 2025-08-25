import { api, handleApiError, handleApiSuccess, cacheManager, cachedRequest } from "./api"

// Types for products
export interface Product {
  id: string
  name: string
  price: number
  stock: number
  description?: string
  category?: string
  images?: string[]
  sku?: string
  status?: 'active' | 'inactive'
  createdAt?: string
  updatedAt?: string
}

export interface CreateProductData {
  name: string
  price: number
  stock: number
  description?: string
  category?: string
  images?: string[]
  sku?: string
  status?: 'active' | 'inactive'
}

export interface UpdateProductData extends Partial<CreateProductData> {
  id: string
}

// Transform backend product data to frontend format
const transformProductData = (product: any): Product => ({
  id: product.id?.toString() || '',
  name: product.name || '',
  price: parseFloat(product.price) || 0,
  stock: product.stock || 0,
  description: product.description || '',
  category: product.category || '',
  images: product.images || [],
  sku: product.sku || '',
  status: product.status || 'active',
  createdAt: product.created_at || '',
  updatedAt: product.updated_at || ''
})

/**
 * Fetch all products for a specific seller
 */
export const getProducts = async (sellerId: string): Promise<Product[]> => {
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

    const transformedProducts = (data.products || []).map(transformProductData)
    return transformedProducts
  } catch (error) {
    handleApiError(error, "Failed to fetch products")
    throw error
  }
}

/**
 * Get a single product by ID
 */
export const getProduct = async (productId: string): Promise<Product> => {
  try {
    const cacheKey = `product-${productId}`
    
    const data = await cachedRequest(
      cacheKey,
      () => api.get(`/products/${productId}`),
      300000 // 5 minutes cache
    )

    return transformProductData(data)
  } catch (error) {
    handleApiError(error, "Failed to fetch product")
    throw error
  }
}

/**
 * Create a new product
 */
export const createProduct = async (sellerId: string, productData: CreateProductData): Promise<Product> => {
  try {
    const data = await api.post(`/products/${sellerId}`, productData)
    
    // Clear related cache
    cacheManager.delete(`products-${sellerId}`)
    
    handleApiSuccess("Product created successfully")
    return transformProductData(data)
  } catch (error) {
    handleApiError(error, "Failed to create product")
    throw error
  }
}

/**
 * Update an existing product
 */
export const updateProduct = async (sellerId: string, productData: UpdateProductData): Promise<Product> => {
  try {
    const { id, ...updateData } = productData
    const data = await api.put(`/products/${id}`, updateData)
    
    // Clear related cache
    cacheManager.delete(`products-${sellerId}`)
    cacheManager.delete(`product-${id}`)
    
    handleApiSuccess("Product updated successfully")
    return transformProductData(data)
  } catch (error) {
    handleApiError(error, "Failed to update product")
    throw error
  }
}

/**
 * Delete a product
 */
export const deleteProduct = async (sellerId: string, productId: string): Promise<void> => {
  try {
    await api.delete(`/products/${productId}`)
    
    // Clear related cache
    cacheManager.delete(`products-${sellerId}`)
    cacheManager.delete(`product-${productId}`)
    
    handleApiSuccess("Product deleted successfully")
  } catch (error) {
    handleApiError(error, "Failed to delete product")
    throw error
  }
}

/**
 * Update product stock
 */
export const updateProductStock = async (productId: string, stock: number): Promise<void> => {
  try {
    await api.put(`/products/${productId}/stock`, { stock })
    
    // Clear related cache
    cacheManager.delete(`product-${productId}`)
    
    handleApiSuccess("Product stock updated successfully")
  } catch (error) {
    handleApiError(error, "Failed to update product stock")
    throw error
  }
}

/**
 * Search products
 */
export const searchProducts = async (sellerId: string, query: string): Promise<Product[]> => {
  try {
    const data = await api.get(`/products/search/${sellerId}?q=${encodeURIComponent(query)}`)
    
    const transformedProducts = (data.products || []).map(transformProductData)
    return transformedProducts
  } catch (error) {
    handleApiError(error, "Failed to search products")
    throw error
  }
}

/**
 * Get products by category
 */
export const getProductsByCategory = async (sellerId: string, category: string): Promise<Product[]> => {
  try {
    const cacheKey = `products-category-${sellerId}-${category}`
    
    const data = await cachedRequest(
      cacheKey,
      () => api.get(`/products/category/${sellerId}/${encodeURIComponent(category)}`),
      300000 // 5 minutes cache
    )

    const transformedProducts = (data.products || []).map(transformProductData)
    return transformedProducts
  } catch (error) {
    handleApiError(error, "Failed to fetch products by category")
    throw error
  }
}

/**
 * Get low stock products
 */
export const getLowStockProducts = async (sellerId: string, threshold: number = 10): Promise<Product[]> => {
  try {
    const data = await api.get(`/products/low-stock/${sellerId}?threshold=${threshold}`)
    
    const transformedProducts = (data.products || []).map(transformProductData)
    return transformedProducts
  } catch (error) {
    handleApiError(error, "Failed to fetch low stock products")
    throw error
  }
}
