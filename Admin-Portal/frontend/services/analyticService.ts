import { toast } from "sonner"
import { api, handleApiError, handleApiSuccess, cacheManager, cachedRequest } from "./api"

export interface OverviewData {
  totalProducts: number;
  totalOrders: number;
  totalUsers: number;
  totalUsersThisMonth: number;
  totalUsersLastMonth: number;
  activeOrders: number;
  totalProfit: number;
  totalProfitThisMonth: number;
  totalProfitLastMonth: number;
}

export interface PopularProductsData {
    id: string;
    name: string;
    queries: number;
    engagement?: string;
}

export interface DailyEngagementData{
    period: string;
    messageCount: number;
}

export interface CustomerInsightsData {
    id: string;
    name: string;
    email: string;
    totalSpent: number;
    totalOrders: number;
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
      activeOrders: response.activeOrders || 0,
      totalProfit: response.totalProfit || 0,
      totalProfitThisMonth: response.totalProfitThisMonth || 0,
      totalProfitLastMonth: response.totalProfitLastMonth || 0,
    };
    return data;
  } catch (error) {
    handleApiError(error);
    return null;
  }
};

const getDailyEngagementByDate = async(sellerId: string, date: string): Promise<DailyEngagementData[] | null> => {
  try {
    const response = await api.get(`/dashboard/messages-by-time-periods/${sellerId}?date=${date}`);
    return response.messagesByTimePeriods || [];
  } catch (error) {
    handleApiError(error);
    return null;
  }
}

const getPopularProducts = async (sellerId: string): Promise<PopularProductsData[] | null> => {
  try {
    const response = await api.get(`/dashboard/popular-products/${sellerId}`);
    const popularProducts = response.popularProducts || [];
    return popularProducts.map((product: any): PopularProductsData => ({
      id: product.productId || '',
      name: product.product.name || '',
      queries: product.mentionCount || 0,
    }));
  } catch (error) {
    handleApiError(error);
    return null;
  }
};

const getAvgResponseTime = async (sellerId: string): Promise<number | null> => {
  try {
    const response = await api.get(`/dashboard/avg-response-time/${sellerId}`);
    return response.avgResponseTime._avg.response_time_ms || 0;
  } catch (error) {
    handleApiError(error);
    return null;
  }
};

const getCustomerByMessageCount = async (sellerId: string, messageCount: number): Promise<number| null> => {
  try {
    const response = await api.get(`/dashboard/customers-with-messages/${sellerId}?limit=${messageCount}`);
    return response.customers || 0;
  } catch (error) {
    handleApiError(error);
    return null;
  }
};

export default {
    fetchOverview,
    getDailyEngagementByDate,
    getPopularProducts,
    getAvgResponseTime,
    getCustomerByMessageCount
}