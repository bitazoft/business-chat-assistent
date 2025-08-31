import express from 'express';
import { addProduct, deleteProduct, updateProduct, getAllProduct, getProductById } from '../controllers/productController.js';
import authenticate from '../middlewares/authMiddleware.js';

const router=express.Router();

// routes
router.post('/add', authenticate, addProduct);
router.post('/delete/:id', authenticate, deleteProduct);
router.post('/update/:id', authenticate, updateProduct);
router.get('/getAll/:id', authenticate, getAllProduct);
router.get('/get/:id', authenticate, getProductById);

export default router;