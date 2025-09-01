import express from 'express'
import { 
  getBusinessDetails, 
  updateBusinessDetails, 
  updateBusinessPassword 
} from '../controllers/businessController.js'

const router = express.Router()

// Get business details for a seller
router.get('/:sellerId', getBusinessDetails)

// Update business details
router.put('/:sellerId', updateBusinessDetails)

// Update business password
router.put('/:sellerId/password', updateBusinessPassword)

export default router
