import { api, cachedRequest, handleApiError } from "./api";

export interface User {
  id: string;
  businessName: string;
  email: string;
  whatsappNumber: string;
  address: string;
  role: "admin" | "user";
  status: "active" | "inactive" | "suspended";
  createdAt: string;
  lastLogin?: string;
  totalOrders: number;
  totalEarned: string;
}

const transformUserData = (user: any): User => {
    const amount = user.totalEarned;
    const date = new Date(user.created_at);
    const formatted = date.toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric"
      });
  
    return {
        id: user.id,
        businessName: user.name,
        email: user.email,
        whatsappNumber: user.phone,
        address: user.address,
        role: user.role,
        status: user.status || "active",
        createdAt: formatted,
        totalOrders: user.totalOrders,
        totalEarned: amount,   
    }
}

export const fetchUsers = async (): Promise<User[]> => {
  try {
    const cacheKey = `users_all`
    
    const data = await cachedRequest(
      cacheKey,
      () => api.get(`/users/getAllUsers`),
      300000 // 5 minutes cache
    )

    console.log('Raw backend data:', data)

    const users: User[] = (data.users || []).map(transformUserData)
    console.log('Transformed orders:', users)

    return users
  } catch (error) {
    handleApiError(error, "Failed to fetch users")
    throw error
  }
};