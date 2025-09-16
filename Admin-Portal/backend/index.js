import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import authRoutes from './routes/authRoutes.js';
import userRoutes from './routes/userRoutes.js';
import productRoutes from './routes/prodcutRoutes.js';
import uploadRoutes from './routes/uploadRoutes.js';
import orderRoutes from './routes/orderRoute.js';
import dashboardRoutes from './routes/dashboardRoutes.js';
import businessRoutes from './routes/businessRoutes.js';
import { dbConnect } from './config/db.js';
import cookieParser from "cookie-parser";

dotenv.config();
dbConnect();

const app = express();

app.use(express.json());
app.use(cookieParser());
app.use(
    cors({
      origin: ["http://localhost:3000","http://47.129.174.20:3000"],
      credentials: true,
    })
  );

app.use('/api/auth', authRoutes);
app.use('/api/users', userRoutes);
app.use('/api/products', productRoutes);
app.use('/api/orders', orderRoutes);
app.use('/api/dashboard', dashboardRoutes);
app.use('/api/business', businessRoutes);
app.use('/api/uploads', uploadRoutes);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Backend running on port ${PORT}`));