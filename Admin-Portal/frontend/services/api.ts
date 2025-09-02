import { toast } from "sonner"

// Base API configuration
export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL,
  TIMEOUT: 10000, // 10 seconds
  Credentials: "include"
}

// Common headers for API requests
export const getHeaders = (additionalHeaders?: Record<string, string>) => ({
  "Content-Type": "application/json",
  ...additionalHeaders,
})

// Common fetch options
export const getFetchOptions = (
  method: string = "GET",
  body?: any,
  additionalHeaders?: Record<string, string>
): RequestInit => ({
  method,
  credentials: "include" as RequestCredentials,
  headers: getHeaders(additionalHeaders),
  ...(body && { body: JSON.stringify(body) }),
})

// Generic API response type
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  message?: string
  error?: string
}

// API error class
export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public response?: Response
  ) {
    super(message)
    this.name = "ApiError"
  }
}

// Generic API request function with error handling
export const apiRequest = async <T = any>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> => {
  const url = `${API_CONFIG.BASE_URL}${endpoint}`
  
  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), API_CONFIG.TIMEOUT)
    
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    })
    
    clearTimeout(timeoutId)

    if (!response.ok) {
      const errorMessage = `HTTP error! status: ${response.status}`
      throw new ApiError(errorMessage, response.status, response)
    }

    // Check if response has content
    const contentType = response.headers.get("content-type")
    if (contentType && contentType.indexOf("application/json") !== -1) {
      return await response.json()
    } else {
      // Return empty object for successful requests without JSON content
      return {} as T
    }
  } catch (error) {
    if (error instanceof ApiError) {
      throw error
    }
    
    if (error instanceof Error) {
      if (error.name === "AbortError") {
        const timeoutError = new ApiError("Request timeout")
        throw timeoutError
      }
      throw new ApiError(error.message)
    }
    
    throw new ApiError("An unknown error occurred")
  }
}

// HTTP method helpers
export const api = {
  get: <T = any>(endpoint: string, headers?: Record<string, string>): Promise<T> =>
    apiRequest<T>(endpoint, getFetchOptions("GET", undefined, headers)),

  post: <T = any>(endpoint: string, data?: any, headers?: Record<string, string>): Promise<T> =>
    apiRequest<T>(endpoint, getFetchOptions("POST", data, headers)),

  put: <T = any>(endpoint: string, data?: any, headers?: Record<string, string>): Promise<T> =>
    apiRequest<T>(endpoint, getFetchOptions("PUT", data, headers)),

  patch: <T = any>(endpoint: string, data?: any, headers?: Record<string, string>): Promise<T> =>
    apiRequest<T>(endpoint, getFetchOptions("PATCH", data, headers)),

  delete: <T = any>(endpoint: string, headers?: Record<string, string>): Promise<T> =>
    apiRequest<T>(endpoint, getFetchOptions("DELETE", undefined, headers)),
}

// Error handler with toast notifications
export const handleApiError = (error: unknown, customMessage?: string): void => {
  console.error("API Error:", error)
  
  if (error instanceof ApiError) {
    const message = customMessage || error.message
    
    switch (error.status) {
      case 401:
        toast.error("Authentication required. Please login again.")
        break
      case 403:
        toast.error("You don't have permission to perform this action.")
        break
      case 404:
        toast.error("Resource not found.")
        break
      case 500:
        toast.error("Server error. Please try again later.")
        break
      default:
        toast.error(message)
    }
  } else {
    toast.error(customMessage || "An unexpected error occurred.")
  }
}

// Success handler with toast notifications
export const handleApiSuccess = (message: string): void => {
  toast.success(message)
}

// Loading state manager
export class LoadingManager {
  private loadingStates = new Map<string, boolean>()
  private listeners = new Map<string, Set<(loading: boolean) => void>>()

  setLoading(key: string, loading: boolean): void {
    this.loadingStates.set(key, loading)
    const keyListeners = this.listeners.get(key)
    if (keyListeners) {
      keyListeners.forEach(listener => listener(loading))
    }
  }

  isLoading(key: string): boolean {
    return this.loadingStates.get(key) || false
  }

  subscribe(key: string, listener: (loading: boolean) => void): () => void {
    if (!this.listeners.has(key)) {
      this.listeners.set(key, new Set())
    }
    this.listeners.get(key)!.add(listener)

    return () => {
      this.listeners.get(key)?.delete(listener)
    }
  }
}

// Global loading manager instance
export const loadingManager = new LoadingManager()

// Utility function for retrying failed requests
export const retryRequest = async <T>(
  requestFn: () => Promise<T>,
  maxRetries: number = 3,
  delay: number = 1000
): Promise<T> => {
  let lastError: Error

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await requestFn()
    } catch (error) {
      lastError = error instanceof Error ? error : new Error("Unknown error")
      
      if (attempt === maxRetries) {
        break
      }
      
      // Wait before retrying (exponential backoff)
      await new Promise(resolve => setTimeout(resolve, delay * attempt))
    }
  }

  throw lastError!
}

// Cache manager for API responses
export class CacheManager {
  private cache = new Map<string, { data: any; timestamp: number; ttl: number }>()

  set(key: string, data: any, ttl: number = 300000): void { // Default 5 minutes TTL
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl
    })
  }

  get(key: string): any | null {
    const cached = this.cache.get(key)
    if (!cached) return null

    if (Date.now() - cached.timestamp > cached.ttl) {
      this.cache.delete(key)
      return null
    }

    return cached.data
  }

  delete(key: string): void {
    this.cache.delete(key)
  }

  clear(): void {
    this.cache.clear()
  }

  // Generate cache key from endpoint and params
  generateKey(endpoint: string, params?: Record<string, any>): string {
    const paramString = params ? JSON.stringify(params) : ""
    return `${endpoint}${paramString}`
  }
}

// Global cache manager instance
export const cacheManager = new CacheManager()

// Cached request wrapper
export const cachedRequest = async <T>(
  key: string,
  requestFn: () => Promise<T>,
  ttl?: number
): Promise<T> => {
  const cached = cacheManager.get(key)
  if (cached) {
    return cached
  }

  const data = await requestFn()
  cacheManager.set(key, data, ttl)
  return data
}
