import { PrismaClient } from "@prisma/client";
const prisma = new PrismaClient();

async function getAllOrders(seller_id) {
    try {
      return await prisma.orders.findMany({
        where: { seller_id },
      });
    } catch (error) {
      throw new ("Error fetching all products", error.message);
    }
  }

export default getAllOrders;