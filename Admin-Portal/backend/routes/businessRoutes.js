import express from 'express'
import { 
  getBusinessDetails, 
  updateBusinessDetails, 
  updateBusinessPassword 
} from '../controllers/businessController.js'
import authorizeRole from '../middlewares/roleMiddleware.js';
import authenticate from '../middlewares/authMiddleware.js';

const router = express.Router()

// Get business details for a seller
router.get('/:sellerId', authenticate, authorizeRole("seller"), getBusinessDetails)

// Update business details
router.put('/:sellerId', authenticate, authorizeRole("seller"), updateBusinessDetails)

// Update business password
router.put('/:sellerId/password', authenticate, authorizeRole("seller"), updateBusinessPassword)

export default router
