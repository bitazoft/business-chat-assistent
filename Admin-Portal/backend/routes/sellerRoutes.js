import express from 'express';
import { registerSeller } from '../controllers/SellerController.js';

const router=express.Router();

router.post('/', registerSeller);

export default router;