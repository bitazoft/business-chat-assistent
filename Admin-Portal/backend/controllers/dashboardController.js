import dashboardService from '../services/dashboardService.js';

const overview = async (req, res) => {
  const seller_id = req.params.id; // Get seller_id from authenticated user
  if (!seller_id) {
    return res.status(400).json({ error: 'User ID not found' });
  }
  try {
    const [
      totalProducts,
      totalOrders,
      totalUsers,
      totalUsersThisMonth,
      totalUsersLastMonth,
      activeOrders,
      totalProfit,
      totalProfitThisMonth,
      totalProfitLastMonth,
      totalMessages,
      totalMessagesThisMonth,
      totalMessagesLastMonth
    ] = await Promise.all([
      dashboardService.getTotalProducts(seller_id),
      dashboardService.getTotalOrders(seller_id),
      dashboardService.getTotalUsers(seller_id),
      dashboardService.getTotalUsersThisMonth(seller_id),
      dashboardService.getTotalUsersLastMonth(seller_id),
      dashboardService.getActiveOrders(seller_id),
      dashboardService.getTotalProfit(seller_id),
      dashboardService.getTotalProfitThisMonth(seller_id),
      dashboardService.getTotalProfitLastMonth(seller_id),
      dashboardService.getTotalMessages(seller_id),
      dashboardService.getTotalMessagesThisMonth(seller_id),
      dashboardService.getTotalMessagesLastMonth(seller_id)
    ]);
    console.log('Dashboard Overview:', {
      totalProducts,
      totalOrders,
      totalUsers,
      totalUsersThisMonth,
      totalUsersLastMonth,
      activeOrders,
      totalProfit,
      totalProfitThisMonth,
      totalProfitLastMonth,
      totalMessages,
      totalMessagesThisMonth,
      totalMessagesLastMonth
    });
    res.json({
      totalProducts,
      totalOrders,
      totalUsers,
      totalUsersThisMonth,
      totalUsersLastMonth,
      activeOrders,
      totalProfit,
      totalProfitThisMonth,
      totalProfitLastMonth,
      totalMessages,
      totalMessagesThisMonth,
      totalMessagesLastMonth
    });
  } catch (error) {
    console.error('Error fetching dashboard overview:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};

const recentOrders = async (req, res) => {
  const seller_id = req.params.id; // Get seller_id from authenticated user
  if (!seller_id) {
    return res.status(400).json({ error: 'User ID not found' });
  }
  try {
    const recentOrders = await dashboardService.getRecentOrders(seller_id);
    console.log('Recent Orders:', recentOrders);
    res.json({ recentOrders });
  } catch (error) {
    console.error('Error fetching recent orders:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};

const topProducts = async (req, res) => {
  const seller_id = req.params.id; // Get seller_id from authenticated user
  if (!seller_id) {
    return res.status(400).json({ error: 'User ID not found' });
  }
  try {
    const topProducts = await dashboardService.getTopProducts(seller_id);
    console.log('Top Products:', topProducts);
    res.json({ topProducts });
  } catch (error) {
    console.error('Error fetching top products:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};

const popularProducts = async (req, res) => {
  const seller_id = req.params.id; // Get seller_id from authenticated user
  if (!seller_id) {
    return res.status(400).json({ error: 'User ID not found' });
  }
  try {
    const popularProducts = await dashboardService.getPopularProducts(seller_id);
    console.log('Popular Products:', popularProducts);
    res.json({ popularProducts });
  } catch (error) {
    console.error('Error fetching popular products:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};

const messagesByTimePeriodsForDate = async (req, res) => {
  const seller_id = req.params.id; // Get seller_id from authenticated user
  const { date } = req.query; // Get date from query parameters

  if (!seller_id) {
    return res.status(400).json({ error: 'User ID not found' });
  }

  if (!date) {
    return res.status(400).json({ error: 'Date is required' });
  }

  try {
    const messagesByTimePeriods = await dashboardService.getMessagesByTimePeriodsForDate(seller_id, date);
    console.log('Messages by Time Periods:', messagesByTimePeriods);
    res.json({ messagesByTimePeriods });
  } catch (error) {
    console.error('Error fetching messages by time periods:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};

const getCustomersWithMoreThanMessages = async (req, res) => {
  const seller_id = req.params.id; // Get seller_id from authenticated user
  const { limit } = req.query; // Get message limit from query parameters (optional)

  if (!seller_id) {
    return res.status(400).json({ error: 'User ID not found' });
  }

  try {
    const messageLimit = limit ? parseInt(limit) : 10; // Default to 10 if not specified
    
    if (isNaN(messageLimit) || messageLimit < 1) {
      return res.status(400).json({ error: 'Invalid message limit. Must be a positive number.' });
    }

    const customers = await dashboardService.getCustomersWithMoreThanMessages(seller_id, messageLimit);
    console.log(`Customers with more than ${messageLimit} messages:`, customers);
    
    res.json({ 
      customers,
      messageLimit,
      totalCustomers: customers.length
    });
  } catch (error) {
    console.error('Error fetching customers with more than specified messages:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};

const getAvgResponseTime = async (req, res) => {
  const seller_id = req.params.id; // Get seller_id from authenticated user
  if (!seller_id) {
    return res.status(400).json({ error: 'User ID not found' });
  }
  try {
    const avgResponseTime = await dashboardService.getAvgResponseTime(seller_id);
    console.log('Average Response Time:', avgResponseTime);
    res.json({ avgResponseTime });
  } catch (error) {
    console.error('Error fetching average response time:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};

export {
  overview,
  recentOrders,
  topProducts,
  popularProducts,
  getCustomersWithMoreThanMessages,
  messagesByTimePeriodsForDate,
  getAvgResponseTime
};