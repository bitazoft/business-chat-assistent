import { api, handleApiError, handleApiSuccess } from "./api"

// Types for business details
export interface BusinessDetails {
  id: number
  shopName: string
  gstNumber?: string
  whatsappNumberId?: string
  ownerName: string
  email: string
  phone?: string
  address?: string
}

export interface UpdateBusinessDetailsData {
  shopName: string
  gstNumber?: string
  whatsappNumberId?: string
  ownerName: string
  email: string
  phone?: string
  address?: string
}

export interface UpdatePasswordData {
  currentPassword: string
  newPassword: string
}

/**
 * Get business details for a seller
 */
export const getBusinessDetails = async (sellerId: string): Promise<BusinessDetails> => {
  try {
    console.log('Fetching business details for seller:', sellerId)
    
    const response = await api.get(`/business/${sellerId}`)

    if (response.success && response.data) {
      console.log('Business details fetched successfully:', response.data)
      return response.data
    }

    throw new Error(response.message || 'Failed to fetch business details')
  } catch (error) {
    console.error('Error fetching business details:', error)
    handleApiError(error, 'Failed to fetch business details')
    throw error
  }
}

/**
 * Update business details
 */
export const updateBusinessDetails = async (
  sellerId: string, 
  data: UpdateBusinessDetailsData
): Promise<BusinessDetails> => {
  try {
    console.log('Updating business details for seller:', sellerId, data)
    
    const response = await api.put(`/business/${sellerId}`, data)

    if (response.success && response.data) {
      console.log('Business details updated successfully:', response.data)
      handleApiSuccess('Business details updated successfully')
      return response.data
    }

    throw new Error(response.message || 'Failed to update business details')
  } catch (error) {
    console.error('Error updating business details:', error)
    handleApiError(error, 'Failed to update business details')
    throw error
  }
}

/**
 * Update business password
 */
export const updateBusinessPassword = async (
  sellerId: string, 
  data: UpdatePasswordData
): Promise<void> => {
  try {
    console.log('Updating password for seller:', sellerId)
    
    const response = await api.put(`/business/${sellerId}/password`, data)

    if (response.success) {
      console.log('Password updated successfully')
      handleApiSuccess('Password updated successfully')
      return
    }

    throw new Error(response.message || 'Failed to update password')
  } catch (error) {
    console.error('Error updating password:', error)
    handleApiError(error, 'Failed to update password')
    throw error
  }
}

// Export default service object
const businessService = {
  getBusinessDetails,
  updateBusinessDetails,
  updateBusinessPassword
}

export default businessService
