import { PrismaClient } from "@prisma/client";
import ProductServiceError from "../utils/errors/productServiceError.js";
const prisma = new PrismaClient();

async function createProduct(data) {
  const { name, description, price, stock, seller_id } = data;

  try {
    return await prisma.products.create({
      data: {
        name,
        description,
        price,
        stock,
        seller_id,
      },
    });
  } catch (error) {
    console.error("Error creating product:", error);
    throw new ProductServiceError("Error creating product", error.message);
  }
}

async function getProductById(id) {
  try {
    return await prisma.products.findUnique({
      where: { id },
    });
  } catch (error) {
    throw new ProductServiceError("Error fetching product by ID", error.message);
  }
}

async function getAllProducts(seller_id) {
  try {
    console.log("Fetching all products for seller:", seller_id);
    return await prisma.products.findMany({
      where: { seller_id },
    });
  } catch (error) {
    throw new ProductServiceError("Error fetching all products", error.message);
  }
}

async function deleteProduct(id) {
  try {
    return await prisma.products.delete({
      where: { id },
    });
  } catch (error) {
    throw new ProductServiceError("Error deleting product", error.message);
  }
}

async function updateProduct(productId, updateData) {
  try {
    return await prisma.products.update({
      where: { id: productId },
      data: updateData,
    });
  } catch (error) {
    throw new ProductServiceError("Error updating product", error.message);
  }
}

export default {
  createProduct,
  getProductById,
  getAllProducts,
  deleteProduct,
  updateProduct,
};
