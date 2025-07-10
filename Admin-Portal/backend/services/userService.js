import { PrismaClient } from '@prisma/client';
const prisma = new PrismaClient();

async function createUser(data) {
  return await prisma.users.create({
    data,
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

