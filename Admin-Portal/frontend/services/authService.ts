import { api, handleApiError, handleApiSuccess, cacheManager, cachedRequest } from "./api"

// Types for auth
export interface User {
  id: string
  sellerId: string
  email: string
  name: string
  phone?: string
  role?: string
  status?: 'active' | 'inactive'
  createdAt?: string
  updatedAt?: string
}

export interface LoginCredentials {
  email: string
  password: string
}

export interface RegisterData {
  email: string
  password: string
  name: string
  phone?: string
}

export interface AuthResponse {
  user: User
  token?: string
  message: string
}

// Transform backend user data to frontend format
const transformUserData = (user: any): User => ({
  id: user.id?.toString() || '',
  sellerId: user.seller_id?.toString() || user.sellerId?.toString() || '',
  email: user.email || '',
  name: user.name || '',
  phone: user.phone || '',
  role: user.role || 'seller',
  status: user.status || 'active',
  createdAt: user.created_at || '',
  updatedAt: user.updated_at || ''
})

/**
 * Login user
 */
export const login = async (credentials: LoginCredentials): Promise<AuthResponse> => {
  try {
    const data = await api.post('/auth/login', credentials)
    
    const authResponse: AuthResponse = {
      user: transformUserData(data.user || data),
      token: data.token,
      message: data.message || "Login successful"
    }
    
    // Store auth token if provided
    if (data.token) {
      localStorage.setItem('authToken', data.token)
    }
    
    // Cache user data
    cacheManager.set('currentUser', authResponse.user, 3600000) // 1 hour
    
    handleApiSuccess("Login successful")
    return authResponse
  } catch (error) {
    handleApiError(error, "Failed to login")
    throw error
  }
}

/**
 * Register new user
 */
export const register = async (userData: RegisterData): Promise<AuthResponse> => {
  try {
    const data = await api.post('/auth/register', userData)
    
    const authResponse: AuthResponse = {
      user: transformUserData(data.user || data),
      token: data.token,
      message: data.message || "Registration successful"
    }
    
    // Store auth token if provided
    if (data.token) {
      localStorage.setItem('authToken', data.token)
    }
    
    // Cache user data
    cacheManager.set('currentUser', authResponse.user, 3600000) // 1 hour
    
    handleApiSuccess("Registration successful")
    return authResponse
  } catch (error) {
    handleApiError(error, "Failed to register")
    throw error
  }
}

/**
 * Logout user
 */
export const logout = async (): Promise<void> => {
  try {
    await api.post('/auth/logout')
    
    // Clear stored data
    localStorage.removeItem('authToken')
    cacheManager.clear()
    
    handleApiSuccess("Logout successful")
  } catch (error) {
    // Even if the API call fails, clear local data
    localStorage.removeItem('authToken')
    cacheManager.clear()
    
    handleApiError(error, "Logout failed, but local session cleared")
    throw error
  }
}

/**
 * Get current user
 */
export const getCurrentUser = async (): Promise<User> => {
  try {
    // Check cache first
    const cachedUser = cacheManager.get('currentUser')
    if (cachedUser) {
      return cachedUser
    }
    
    const data = await api.get('/auth/me')
    const user = transformUserData(data.user || data)
    
    // Cache user data
    cacheManager.set('currentUser', user, 3600000) // 1 hour
    
    return user
  } catch (error) {
    // Clear cache and storage on auth failure
    cacheManager.delete('currentUser')
    localStorage.removeItem('authToken')
    
    handleApiError(error, "Failed to get user info")
    throw error
  }
}

/**
 * Update user profile
 */
export const updateProfile = async (userData: Partial<User>): Promise<User> => {
  try {
    const data = await api.put('/auth/profile', userData)
    const user = transformUserData(data.user || data)
    
    // Update cache
    cacheManager.set('currentUser', user, 3600000) // 1 hour
    
    handleApiSuccess("Profile updated successfully")
    return user
  } catch (error) {
    handleApiError(error, "Failed to update profile")
    throw error
  }
}

/**
 * Change password
 */
export const changePassword = async (currentPassword: string, newPassword: string): Promise<void> => {
  try {
    await api.put('/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword
    })
    
    handleApiSuccess("Password changed successfully")
  } catch (error) {
    handleApiError(error, "Failed to change password")
    throw error
  }
}

/**
 * Refresh auth token
 */
export const refreshToken = async (): Promise<string> => {
  try {
    const data = await api.post('/auth/refresh')
    
    if (data.token) {
      localStorage.setItem('authToken', data.token)
      return data.token
    }
    
    throw new Error("No token received")
  } catch (error) {
    // Clear auth data on refresh failure
    localStorage.removeItem('authToken')
    cacheManager.delete('currentUser')
    
    handleApiError(error, "Failed to refresh token")
    throw error
  }
}

/**
 * Request password reset
 */
export const requestPasswordReset = async (email: string): Promise<void> => {
  try {
    await api.post('/auth/forgot-password', { email })
    
    handleApiSuccess("Password reset email sent")
  } catch (error) {
    handleApiError(error, "Failed to send password reset email")
    throw error
  }
}

/**
 * Reset password with token
 */
export const resetPassword = async (token: string, newPassword: string): Promise<void> => {
  try {
    await api.post('/auth/reset-password', {
      token,
      new_password: newPassword
    })
    
    handleApiSuccess("Password reset successfully")
  } catch (error) {
    handleApiError(error, "Failed to reset password")
    throw error
  }
}

/**
 * Check if user is authenticated
 */
export const isAuthenticated = (): boolean => {
  const token = localStorage.getItem('authToken')
  const cachedUser = cacheManager.get('currentUser')
  
  return !!(token && cachedUser)
}

/**
 * Get stored auth token
 */
export const getAuthToken = (): string | null => {
  return localStorage.getItem('authToken')
}
