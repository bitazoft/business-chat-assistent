import { toast } from "sonner"
import { api, handleApiError, handleApiSuccess, cacheManager, cachedRequest } from "./api"


export interface OverviewData {
  totalProducts: number;
  totalOrders: number;
  totalUsers: number;
  totalUsersThisMonth: number;
  totalUsersLastMonth: number;
  totalUsersToday: number;
  activeOrders: number;
  totalProfit: number;
  totalProfitThisMonth: number;
  totalProfitLastMonth: number;
  totalProfitToday: number;
}

export interface TopProductsData {
  productId: string;
  productName: string;
  totalSales: number;
  totalQuantity: number;
}

export interface RecentOrdersData {
  orderId: string;
  customerName: string;
  totalAmount?: number; // Made optional to handle undefined cases
  createdAt: Date;
  status: string;
}

const fetchOverview = async (sellerId : string): Promise<OverviewData | null> => {
  try {
    const response = await api.get(`/dashboard/overview/${sellerId}`);
    // Backend returns the data directly for overview
    const data: OverviewData = {
      totalProducts: response.totalProducts || 0,
      totalOrders: response.totalOrders || 0,
      totalUsers: response.totalUsers || 0,
      totalUsersThisMonth: response.totalUsersThisMonth || 0,
      totalUsersLastMonth: response.totalUsersLastMonth || 0,
      totalUsersToday: response.totalUsersToday || 0,
      activeOrders: response.activeOrders || 0,
      totalProfit: response.totalProfit || 0,
      totalProfitThisMonth: response.totalProfitThisMonth || 0,
      totalProfitLastMonth: response.totalProfitLastMonth || 0,
      totalProfitToday: response.totalProfitToday || 0,
    };
    return data;
  } catch (error) {
    handleApiError(error);
    return null;
  }
};

const fetchTopProducts = async (sellerId: string): Promise<TopProductsData[] | null> => {
  try {
    const response = await api.get(`/dashboard/top-products/${sellerId}`);
    // Backend returns { topProducts: [...] }, so we need to extract the array
    const topProducts = response.topProducts || [];
    return topProducts.map((product: any): TopProductsData => ({
      productId: product.product.id || '',
      productName: product.product.name || '',
      totalSales: product.totalSalesQuantity || 0,
      totalQuantity: product.product.stock || 0,
    }));
  } catch (error) {
    handleApiError(error);
    return null;
  }
};

const fetchRecentOrders = async (sellerId: string): Promise<RecentOrdersData[] | null> => {
  try {
    const response = await api.get(`/dashboard/recent-orders/${sellerId}`);
    // Backend returns { recentOrders: [...] }, so we need to extract the array
    const recentOrders = response.recentOrders || [];
    return recentOrders.map((order: any): RecentOrdersData => ({
      orderId: order.id || '',
      customerName: order.customers.name || '',
      totalAmount: order.total_amount,
      createdAt: new Date(order.created_at),
      status: order.status || '',
    }));
  } catch (error) {
    handleApiError(error);
    return null;
  }
};

export default {
  fetchOverview,
  fetchTopProducts,
  fetchRecentOrders
};