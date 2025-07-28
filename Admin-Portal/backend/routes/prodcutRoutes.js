import express from 'express';
import { addProduct, deleteProduct, updateProduct, getAllProduct, getProductById } from '../controllers/productController.js';

const router=express.Router();

router.post('/add', addProduct);
router.post('/delete/:id', deleteProduct);
router.post('/update/:id', updateProduct);
router.get('/getAll/:id', getAllProduct);
router.get('/get/:id', getProductById);

export default router;