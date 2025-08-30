import express from "express";
import authenticate from "../middlewares/authMiddleware.js";
import uploadFile from "../controllers/uploadController.js";

const router = express.Router()

router.post('/image', authenticate, uploadFile);

export default router;