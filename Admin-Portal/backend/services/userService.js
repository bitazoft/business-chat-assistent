import { PrismaClient } from '@prisma/client';
const prisma = new PrismaClient();

async function createUser(data) {
  const {
    name,
    email,
    phone,
    password,
    role,
    address,
    shop_name,
    whatsapp_number_id,
  } = data;

  try {
    return await prisma.users.create({
      data: {
        name,
        email,
        phone,
        password,
        role,
        address,
  
        ...(role === "seller" && {
          seller_profiles: {
            create: {
              shop_name,
              whatsapp_number_id,
            },
          },
        }),
      },
      include: {
        seller_profiles: true,
      },
    });
  } catch (error) {
    throw new Error(`Error creating user: ${error.message}`);
  }
}


async function getUserByEmail(email) {
  try {
    return await prisma.users.findUnique({
      where: { email },
      include: {
        seller_profiles: true,
      },
    });
  } catch (error) {
    throw new Error(`Error fetching user by email: ${error.message}`);
  }
}

async function getAllUsers() {
  try {
    return await prisma.users.findMany();
  } catch (error) {
    throw new Error(`Error fetching all users: ${error.message}`);
  }
}

async function deleteUser(id) {
  try {
    return await prisma.users.delete({
      where: { id },
    });
  } catch (error) {
    throw new Error(`Error deleting user: ${error.message}`);
  }
}

export default {
  createUser,
  getUserByEmail,
  getAllUsers,
  deleteUser,
};