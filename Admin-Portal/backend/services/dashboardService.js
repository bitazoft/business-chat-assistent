import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

function getTotalProducts(seller_id) {
  return prisma.products.count({
    where: {
      seller_id: parseInt(seller_id)
    }
  });
}

function getTotalOrders(seller_id) {
  return prisma.orders.count({
    where: {
      seller_id: parseInt(seller_id)
    }
  });
}

function getActiveOrders(seller_id) {
return prisma.orders.count({
    where: {
        seller_id: parseInt(seller_id),
        status: {
            in: ['pending', 'processing', 'shipped']
        }
    }
});
}

function getTotalUsers(seller_id) {
  return prisma.chat_logs.groupBy({
    by: ['customer_id'],
    where: {
      seller_id: parseInt(seller_id),
      customer_id: {
        not: null
      }
    },
    _count: {
      customer_id: true
    }
  }).then(result => result.length);
}

function getTotalMessagesLastMonth(seller_id) {
  const now = new Date();
  const startOfLastMonth = new Date(now.getFullYear(), now.getMonth() - 1, 1);
  const endOfLastMonth = new Date(now.getFullYear(), now.getMonth(), 0, 23, 59, 59, 999);

  return prisma.chat_logs.count({
    where: {
      seller_id: parseInt(seller_id),
      timestamp: {
        gte: startOfLastMonth,
        lte: endOfLastMonth
      }
    }
  });
}

function getTotalMessagesThisMonth(seller_id) {
  const now = new Date();
  const startOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);
  const endOfMonth = new Date(now.getFullYear(), now.getMonth() + 1, 0, 23, 59, 59, 999);

  return prisma.chat_logs.count({
    where: {
      seller_id: parseInt(seller_id),
      timestamp: {
        gte: startOfMonth,
        lte: endOfMonth
      }
    }
  });
}

function getTotalMessages(seller_id) {
  return prisma.chat_logs.count({
    where: {
      seller_id: parseInt(seller_id)
    }
  });
}



function getTotalUsersThisMonth(seller_id) {
  const now = new Date();
  const startOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);
  const endOfMonth = new Date(now.getFullYear(), now.getMonth() + 1, 0, 23, 59, 59, 999);

  return prisma.chat_logs.groupBy({
    by: ['customer_id'],
    where: {
      seller_id: parseInt(seller_id),
      customer_id: {
        not: null
      },
      timestamp: {
        gte: startOfMonth,
        lte: endOfMonth
      }
    },
    _count: {
      customer_id: true
    }
  }).then(result => result.length);
}

function getTotalUsersLastMonth(seller_id) {
  const now = new Date();
  const startOfLastMonth = new Date(now.getFullYear(), now.getMonth() - 1, 1);
  const endOfLastMonth = new Date(now.getFullYear(), now.getMonth(), 0, 23, 59, 59, 999);

  return prisma.chat_logs.groupBy({
    by: ['customer_id'],
    where: {
      seller_id: parseInt(seller_id),
      customer_id: {
        not: null
      },
      timestamp: {
        gte: startOfLastMonth,
        lte: endOfLastMonth
      }
    },
    _count: {
      customer_id: true
    }
  }).then(result => result.length);
}


function getRecentOrders(seller_id) {
  return prisma.orders.findMany({
    where: {
      seller_id: parseInt(seller_id)
    },
    include: {
      customers: true
    },
    orderBy: {
      created_at: 'desc'
    },
    take: 3
  });
}

function getTopProducts(seller_id) {
  return prisma.order_items.groupBy({
    by: ['product_id'],
    where: {
      orders: {
        seller_id: parseInt(seller_id)
      }
    },
    _count: {
      product_id: true
    },
    _sum: {
      quantity: true
    },
    orderBy: {
      _count: {
        product_id: 'desc'
      }
    },
    take: 3
  }).then(async (groupedItems) => {
    // Get product details for each grouped item
    const productDetails = await Promise.all(
      groupedItems.map(async (item) => {
        const product = await prisma.products.findUnique({
          where: { id: item.product_id }
        });
        return {
          product: product,
          orderCount: item._count.product_id,
          totalSalesQuantity: item._sum.quantity
        };
      })
    );
    return productDetails;
  });
}

function getTotalProfit(seller_id) {
  return prisma.orders.aggregate({
    _sum: {
      total_amount: true
    },
    where: {
      seller_id: parseInt(seller_id)
    }
  }).then(result => result._sum.total_amount || 0);
}

function getTotalProfitThisMonth(seller_id) {
  const now = new Date();
  const startOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);
  const endOfMonth = new Date(now.getFullYear(), now.getMonth() + 1, 0, 23, 59, 59, 999);

  return prisma.orders.aggregate({
    _sum: {
      total_amount: true
    },
    where: {
      seller_id: parseInt(seller_id),
      created_at: {
        gte: startOfMonth,
        lte: endOfMonth
      }
    }
  }).then(result => result._sum.total_amount || 0);
}

function getTotalProfitLastMonth(seller_id) {
  const now = new Date();
  const startOfLastMonth = new Date(now.getFullYear(), now.getMonth() - 1, 1);
  const endOfLastMonth = new Date(now.getFullYear(), now.getMonth(), 0, 23, 59, 59, 999);

  return prisma.orders.aggregate({
    _sum: {
      total_amount: true
    },
    where: {
      seller_id: parseInt(seller_id),
      created_at: {
        gte: startOfLastMonth,
        lte: endOfLastMonth
      }
    }
  }).then(result => result._sum.total_amount || 0);
}

function getPopularProducts(seller_id) {
  return prisma.chat_logs.findMany({
    where: {
      seller_id: parseInt(seller_id),
      entities: {
        not: null
      }
    },
    select: {
      entities: true
    }
  }).then(async (chatLogs) => {
    // Extract product ID mentions from entities
    const productMentions = new Map();
    
    chatLogs.forEach(log => {
      try {
        if (log.entities) {
          // Parse entities JSON string
          // const entities = JSON.parse(log.entities);
          const entities = typeof log.entities === 'string' ? JSON.parse(log.entities) : log.entities;
          
          // Check for product_id directly in entities
          if (entities.product_id) {
            const productId = parseInt(entities.product_id);
            if (!isNaN(productId)) {
              productMentions.set(productId, (productMentions.get(productId) || 0) + 1);
            }
          }
          
          // Check if entities has products array with IDs
          else if (entities.entities && entities.entities.products && Array.isArray(entities.entities.products)) {
            entities.entities.products.forEach(productRef => {
              const productId = parseInt(productRef);
              if (!isNaN(productId)) {
                productMentions.set(productId, (productMentions.get(productId) || 0) + 1);
              }
            });
          }
          
          // Also check for direct product references with IDs
          else if (entities.products && Array.isArray(entities.products)) {
            entities.products.forEach(productRef => {
              const productId = parseInt(productRef);
              if (!isNaN(productId)) {
                productMentions.set(productId, (productMentions.get(productId) || 0) + 1);
              }
            });
          }
        }
      } catch (error) {
        // Skip invalid JSON entities
        console.log('Error parsing entities:', error);
      }
    });
    
    // Sort by mention count and get top 3
    const sortedMentions = Array.from(productMentions.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);
    
    // Get actual product details by product IDs
    const productDetails = await Promise.all(
      sortedMentions.map(async ([productId, mentionCount]) => {
        // Find product by ID
        const product = await prisma.products.findUnique({
          where: {
            id: productId,
            seller_id: parseInt(seller_id)
          }
        });
        
        return {
          product: product,
          mentionCount: mentionCount,
          productId: productId
        };
      })
    );
    
    return productDetails;
  });
}

function getMessagesByTimePeriodsForDate(seller_id, date) {
  // Parse the input date (expecting YYYY-MM-DD format)
  const targetDate = new Date(date);
  if (isNaN(targetDate.getTime())) {
    throw new Error('Invalid date format. Please use YYYY-MM-DD.');
  }

  // Define time periods
  const timePeriods = [
    { label: '12AM-6AM', start: 0, end: 6 },
    { label: '6AM-9AM', start: 6, end: 9 },
    { label: '9AM-12PM', start: 9, end: 12 },
    { label: '12PM-3PM', start: 12, end: 15 },
    { label: '3PM-6PM', start: 15, end: 18 },
    { label: '6PM-9PM', start: 18, end: 21 },
    { label: '9PM-12AM', start: 21, end: 24 }
  ];

  return Promise.all(
    timePeriods.map(async (period) => {
      // Create start and end datetime for the period
      const startTime = new Date(targetDate);
      startTime.setHours(period.start, 0, 0, 0);
      
      const endTime = new Date(targetDate);
      endTime.setHours(period.end, 0, 0, 0);
      
      // Count messages in this time period
      const messageCount = await prisma.chat_logs.count({
        where: {
          seller_id: parseInt(seller_id),
          timestamp: {
            gte: startTime,
            lt: endTime
          }
        }
      });

      return {
        period: period.label,
        startHour: period.start,
        endHour: period.end,
        messageCount: messageCount,
        startTime: startTime.toISOString(),
        endTime: endTime.toISOString()
      };
    })
  );
}

function getAvgResponseTime(seller_id) {

  return prisma.chat_logs.aggregate({
    where: {
      seller_id: parseInt(seller_id),
    },
    _avg: {
      response_time_ms: true
    }
  });
}

async function getCustomersWithMoreThanMessages(seller_id, messageLimit = 10) {
  try {
    // Get message count per customer
    const customerMessageCounts = await prisma.chat_logs.groupBy({
      by: ['customer_id'],
      where: {
        seller_id: parseInt(seller_id),
        customer_id: {
          not: null
        }
      },
      _count: {
        customer_id: true
      },
      having: {
        customer_id: {
          _count: {
            gt: messageLimit
          }
        }
      }
    });

    // Get customer details for those with more than the specified limit
    const customerIds = customerMessageCounts.map(item => item.customer_id);
    
    if (customerIds.length === 0) {
      return [];
    }

    const customersWithDetails = await prisma.customers.findMany({
      where: {
        id: {
          in: customerIds
        }
      },
      select: {
        id: true,
        name: true,
        email: true,
        number1: true,
        number2: true,
        created_at: true
      }
    });

    // Combine customer details with message counts
    const result = customersWithDetails.map(customer => {
      const messageCount = customerMessageCounts.find(
        item => item.customer_id === customer.id
      )?._count?.customer_id || 0;

      return {
        ...customer,
        messageCount
      };
    });

    // Sort by message count in descending order
    return result.sort((a, b) => b.messageCount - a.messageCount);

  } catch (error) {
    console.error('Error fetching customers with more than specified messages:', error);
    throw error;
  }
}



export default{ 
  getTotalProducts, 
  getTotalOrders, 
  getTotalUsers, 
  getTotalUsersThisMonth,
  getTotalUsersLastMonth,
  getRecentOrders,
  getTopProducts,
  getActiveOrders,
  getTotalProfit,
  getTotalProfitThisMonth,
  getTotalProfitLastMonth,
  getTotalMessages,
  getTotalMessagesThisMonth,
  getTotalMessagesLastMonth,
  getPopularProducts,
  getMessagesByTimePeriodsForDate,
  getAvgResponseTime,
  getCustomersWithMoreThanMessages
};