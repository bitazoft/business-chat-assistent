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

  return await prisma.users.create({
    data: {
      name,
      email,
      phone,
      password,
      role,
      address,

      ...(role === "seller" && {
        seller_profile: {
          create: {
            shop_name,
            whatsapp_number_id,
          },
        },
      }),
    },
    include: {
      seller_profile: true,
    },
  });
}


async function getUserByEmail(email) {
  return await prisma.users.findUnique({
    where: { email },
  });
}

async function getAllUsers() {
  return await prisma.users.findMany();
}

async function deleteUser(id) {
  return await prisma.users.delete({
    where: { id },
  });
}

export default {
  createUser,
  getUserByEmail,
  getAllUsers,
  deleteUser,
};

