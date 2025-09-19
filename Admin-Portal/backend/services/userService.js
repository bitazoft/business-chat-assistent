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
  
        ...(role === seller && {
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
    return await prisma.users.findMany({
      select: {
        id: true,
        email: true,
        name: true,
        phone: true,
        role: true,
        address: true,
        created_at: true,
        seller_profiles: {
          select: {
            shop_name: true,
          }
        }
      }
    });    
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

async function usersWithStats(){
  const users = await getAllUsers();

  return Promise.all(
    users.map(async (user)=>{
      const ordersSummary = await prisma.orders.groupBy({
        where: { seller_id: user.id },
        by: ['seller_id'],
        _count: {
          id: true, 
        },
        _sum: {
          total_amount: true,
        },
      });

      const summary = ordersSummary.map(o => ({
        totalOrders: o._count.id,
        totalEarned: o._sum.total_amount
      }));  

      const { seller_profiles, ...rest } = user;

      return {
        ...rest,
        name: seller_profiles?.shop_name || user.name,
        totalOrders: summary[0]?.totalOrders || 0,
        totalEarned: summary[0]?.totalEarned || 0,
      };
    })
  )
}

export default {
  createUser,
  getUserByEmail,
  getAllUsers,
  deleteUser,
  usersWithStats
};