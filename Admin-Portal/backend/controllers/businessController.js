import { PrismaClient } from '@prisma/client'
import bcrypt from 'bcrypt'

const prisma = new PrismaClient()

// Get business details for a seller
export const getBusinessDetails = async (req, res) => {
  try {
    const { sellerId } = req.params

    if (!sellerId) {
      return res.status(400).json({
        success: false,
        message: 'Seller ID is required'
      })
    }

    // Get seller profile with user details
    const sellerProfile = await prisma.seller_profiles.findUnique({
      where: {
        id: parseInt(sellerId)
      },
      include: {
        users: {
          select: {
            id: true,
            name: true,
            email: true,
            phone: true,
            address: true
          }
        }
      }
    })

    if (!sellerProfile) {
      return res.status(404).json({
        success: false,
        message: 'Business not found'
      })
    }

    const businessDetails = {
      id: sellerProfile.id,
      shopName: sellerProfile.shop_name,
      gstNumber: sellerProfile.gst_number,
      whatsappNumberId: sellerProfile.whatsapp_number_id,
      ownerName: sellerProfile.users.name,
      email: sellerProfile.users.email,
      phone: sellerProfile.users.phone,
      address: sellerProfile.users.address
    }

    res.status(200).json({
      success: true,
      data: businessDetails,
      message: 'Business details fetched successfully'
    })

  } catch (error) {
    console.error('Error fetching business details:', error)
    res.status(500).json({
      success: false,
      message: 'Internal server error',
      error: error.message
    })
  }
}

// Update business details
export const updateBusinessDetails = async (req, res) => {
  try {
    const { sellerId } = req.params
    const {
      shopName,
      gstNumber,
      whatsappNumberId,
      ownerName,
      email,
      phone,
      address
    } = req.body

    if (!sellerId) {
      return res.status(400).json({
        success: false,
        message: 'Seller ID is required'
      })
    }

    // Validate required fields
    if (!shopName || !ownerName || !email) {
      return res.status(400).json({
        success: false,
        message: 'Shop name, owner name, and email are required'
      })
    }

    // Check if seller profile exists
    const existingProfile = await prisma.seller_profiles.findUnique({
      where: {
        id: parseInt(sellerId)
      },
      include: {
        users: true
      }
    })

    if (!existingProfile) {
      return res.status(404).json({
        success: false,
        message: 'Business not found'
      })
    }

    // Check if email is being changed and if it already exists for another user
    if (email !== existingProfile.users.email) {
      const existingUser = await prisma.users.findUnique({
        where: { email }
      })

      if (existingUser && existingUser.id !== existingProfile.user_id) {
        return res.status(400).json({
          success: false,
          message: 'Email address already in use'
        })
      }
    }

    // Use transaction to update both user and seller profile
    const result = await prisma.$transaction(async (tx) => {
      // Update user details
      const updatedUser = await tx.users.update({
        where: {
          id: existingProfile.user_id
        },
        data: {
          name: ownerName,
          email: email,
          phone: phone || existingProfile.users.phone,
          address: address || existingProfile.users.address
        }
      })

      // Update seller profile
      const updatedProfile = await tx.seller_profiles.update({
        where: {
          id: parseInt(sellerId)
        },
        data: {
          shop_name: shopName,
          gst_number: gstNumber || existingProfile.gst_number,
          whatsapp_number_id: whatsappNumberId || existingProfile.whatsapp_number_id
        }
      })

      return { updatedUser, updatedProfile }
    })

    const businessDetails = {
      id: result.updatedProfile.id,
      shopName: result.updatedProfile.shop_name,
      gstNumber: result.updatedProfile.gst_number,
      whatsappNumberId: result.updatedProfile.whatsapp_number_id,
      ownerName: result.updatedUser.name,
      email: result.updatedUser.email,
      phone: result.updatedUser.phone,
      address: result.updatedUser.address
    }

    res.status(200).json({
      success: true,
      data: businessDetails,
      message: 'Business details updated successfully'
    })

  } catch (error) {
    console.error('Error updating business details:', error)
    res.status(500).json({
      success: false,
      message: 'Internal server error',
      error: error.message
    })
  }
}

// Update business password
export const updateBusinessPassword = async (req, res) => {
  try {
    const { sellerId } = req.params
    const { currentPassword, newPassword } = req.body

    if (!sellerId || !currentPassword || !newPassword) {
      return res.status(400).json({
        success: false,
        message: 'All fields are required'
      })
    }

    if (newPassword.length < 6) {
      return res.status(400).json({
        success: false,
        message: 'New password must be at least 6 characters long'
      })
    }

    // Get seller profile with user details
    const sellerProfile = await prisma.seller_profiles.findUnique({
      where: {
        id: parseInt(sellerId)
      },
      include: {
        users: true
      }
    })

    if (!sellerProfile) {
      return res.status(404).json({
        success: false,
        message: 'Business not found'
      })
    }

    // Verify current password
    const isCurrentPasswordValid = await bcrypt.compare(currentPassword, sellerProfile.users.password)
    if (!isCurrentPasswordValid) {
      return res.status(400).json({
        success: false,
        message: 'Current password is incorrect'
      })
    }

    // Hash new password
    const hashedNewPassword = await bcrypt.hash(newPassword, 10)

    // Update password
    await prisma.users.update({
      where: {
        id: sellerProfile.user_id
      },
      data: {
        password: hashedNewPassword
      }
    })

    res.status(200).json({
      success: true,
      message: 'Password updated successfully'
    })

  } catch (error) {
    console.error('Error updating password:', error)
    res.status(500).json({
      success: false,
      message: 'Internal server error',
      error: error.message
    })
  }
}
