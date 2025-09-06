import { PrismaClient } from "@prisma/client";
import ProductServiceError from "../utils/errors/productServiceError.js";
import ProductResponse from "../models/productResponseModel.js";
const prisma = new PrismaClient();

async function createProduct(data) {
  const { name, description, price, stock, images, seller_id } = data;

  try {
    return await prisma.products.create({
      data: {
        name,
        description,
        price,
        stock,
        item_img: {
          create: images.map((img) => ({
            image_url: img.url,
            is_main: img.isMain,
          })),
        },
        seller_id,
      },
      include: {
        item_img: true,
      },
    });
  } catch (error) {
    console.error("Error creating product:", error);
    throw new ProductServiceError("Error creating product", error.message);
  }
}

async function getProductById(id) {
  try {
    const product = await prisma.products.findUnique({
      where: { id },
      include: {
        item_img: true,
      },
    });

    if (!product) return null;

    return new ProductResponse({
      id: product.id,
      name: product.name,
      price: product.price,
      stock: product.stock,
      category: product.category,
      status: product.status,
      description: product.description,
      images: product.item_img.map((img) => ({
        id: img.id,
        url: img.image_url,
        isMain: img.is_main, // Prisma snake_case → DTO camelCase
      })),
    });
  } catch (error) {
    throw new ProductServiceError("Error fetching product by ID", error.message);
  }
}

async function getAllProducts(seller_id) {
  try {
    console.log("Fetching all products for seller:", seller_id);
    const res = await prisma.products.findMany({
      where: { seller_id },
      include: {
        item_img: true,
      },
    });

    return res.map(
      (product) =>
        new ProductResponse({
          id: product.id,
          name: product.name,
          price: product.price,
          stock: product.stock,
          category: product.category,
          status: product.status,
          description: product.description,
          images: product.item_img.map((img) => ({
            id: img.id,
            url: img.image_url,
            isMain: img.is_main,
          })),
        })
    );
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
  const { name, description, price, stock, images } = updateData;

  try {
    return await prisma.$transaction(async (tx) => {
      for (const img of images) {
        if ("id" in img) {
          await tx.item_img.update({
            where: { id: img.id },
            data: { image_url: img.url, is_main: img.isMain },
          });
        }else{
          await tx.item_img.create({
            data: { image_url: img.url, is_main: img.isMain, product_id: productId },
          });
        }
      }
    
      return tx.products.update({
        where: { id: productId },
        data: {
          name,
          description,
          price,
          stock,
        },
        include: { item_img: true },
      });
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
