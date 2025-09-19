import express from 'express';
import { addProduct, deleteProduct, updateProduct, getAllProduct, getProductById } from '../controllers/productController.js';
import authenticate from '../middlewares/authMiddleware.js';
import authorizeRole from '../middlewares/roleMiddleware.js';

const router=express.Router();

router.post('/add', authenticate, authorizeRole("seller"), addProduct);
router.post('/delete/:id', authenticate, authorizeRole("seller"), deleteProduct);
router.post('/update/:id', authenticate, authorizeRole("seller"), updateProduct);
router.get('/getAll/:id', authenticate, authorizeRole("seller"), getAllProduct);
router.get('/get/:id', authenticate, authorizeRole("seller"), getProductById);

export default router;