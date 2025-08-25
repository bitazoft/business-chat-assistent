# Services Documentation

This folder contains all the API service functions for the frontend application. The services are organized by functionality and provide a clean abstraction layer for API calls.

## Structure

```
services/
├── api.ts              # Base API configuration and utilities
├── authService.ts      # Authentication related APIs
├── orderService.ts     # Order management APIs
├── productService.ts   # Product management APIs
└── index.ts           # Central export file
```

## Usage

### Importing Services

You can import services individually or from the central index:

```typescript
// Import specific services
import { fetchOrders, updateOrderStatus } from '@/services/orderService'
import { login, getCurrentUser } from '@/services/authService'

// Or import from central index
import { fetchOrders, updateOrderStatus, login, getCurrentUser } from '@/services'
```

### Order Service Example

Replace the existing fetch calls in your components:

**Before:**
```typescript
const fetchOrders = async () => {
  if (!user?.sellerId) {
    toast.error("User not authenticated")
    setLoading(false)
    return
  }

  try {
    const response = await fetch(`http://localhost:7001/api/orders/${user.sellerId}`, {
      method: "GET",
      credentials: "include",
      headers: {
        "Content-Type": "application/json"
      },
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    // ... transformation logic
    setOrders(transformedOrders)
  } catch (error) {
    console.error("Error fetching orders:", error)
    toast.error("Failed to fetch orders")
  } finally {
    setLoading(false)
  }
}
```

**After:**
```typescript
import { fetchOrders as getOrders } from '@/services/orderService'

const fetchOrders = async () => {
  if (!user?.sellerId) {
    setLoading(false)
    return
  }

  try {
    const orders = await getOrders(user.sellerId)
    setOrders(orders)
  } catch (error) {
    // Error handling is already done in the service
  } finally {
    setLoading(false)
  }
}
```

### Product Service Example

```typescript
import { getProducts, createProduct, updateProduct, deleteProduct } from '@/services/productService'

// Fetch products
const loadProducts = async () => {
  try {
    const products = await getProducts(sellerId)
    setProducts(products)
  } catch (error) {
    // Error is already handled in service
  }
}

// Create a new product
const handleCreateProduct = async (productData) => {
  try {
    const newProduct = await createProduct(sellerId, productData)
    setProducts(prev => [...prev, newProduct])
  } catch (error) {
    // Error is already handled in service
  }
}
```

### Auth Service Example

```typescript
import { login, getCurrentUser, logout } from '@/services/authService'

// Login
const handleLogin = async (email, password) => {
  try {
    const authResponse = await login({ email, password })
    setUser(authResponse.user)
    // Redirect to dashboard
  } catch (error) {
    // Error is already handled in service
  }
}

// Get current user
const loadCurrentUser = async () => {
  try {
    const user = await getCurrentUser()
    setUser(user)
  } catch (error) {
    // Redirect to login
  }
}
```

## Features

### 1. Error Handling
All services include comprehensive error handling with toast notifications:
- Automatic error messages
- HTTP status code handling
- Authentication error redirects

### 2. Caching
API responses are cached to improve performance:
- 5-minute cache for orders and products
- 1-hour cache for user data
- Automatic cache invalidation on updates

### 3. Loading States
Services work with the global loading manager:
```typescript
import { loadingManager } from '@/services/api'

// Subscribe to loading state
useEffect(() => {
  const unsubscribe = loadingManager.subscribe('orders', setLoading)
  return unsubscribe
}, [])
```

### 4. Retry Logic
Failed requests are automatically retried with exponential backoff:
```typescript
import { retryRequest } from '@/services/api'

const data = await retryRequest(() => api.get('/orders'), 3, 1000)
```

### 5. Type Safety
All services are fully typed with TypeScript interfaces:
```typescript
interface Order {
  id: string
  customer: string
  phone: string
  // ... other properties
}
```

## Configuration

### Environment Variables
Create a `.env.local` file and set:
```
NEXT_PUBLIC_API_BASE_URL=http://localhost:7001/api
```

### API Base URL
Update the base URL in `services/api.ts`:
```typescript
export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:7001/api",
  TIMEOUT: 10000,
}
```

## Migration Guide

To migrate existing components to use these services:

1. **Import the service functions** instead of using direct fetch calls
2. **Remove manual error handling** - it's handled in the services
3. **Remove data transformation logic** - it's done in the services
4. **Update component state management** to work with the service responses
5. **Remove manual toast notifications** for API operations

### Example Migration

**Before:**
```typescript
const updateOrderStatus = async (orderId: string, newStatus: string) => {
  try {
    const response = await fetch(`http://localhost:7001/api/orders/${orderId}`, {
      method: "PUT",
      credentials: "include",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ status: newStatus.toLowerCase() }),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    await fetchOrders()
    toast.success(`Order status updated to ${newStatus}`)
  } catch (error) {
    console.error("Error updating order status:", error)
    toast.error("Failed to update order status")
  }
}
```

**After:**
```typescript
import { updateOrderStatus as updateStatus, fetchOrders as getOrders } from '@/services/orderService'

const updateOrderStatus = async (orderId: string, newStatus: string) => {
  try {
    await updateStatus(orderId, newStatus)
    
    // Refresh orders
    const orders = await getOrders(user.sellerId)
    setOrders(orders)
  } catch (error) {
    // Error handling is done in the service
  }
}
```

## Best Practices

1. **Always handle errors** in your components, even though services provide default error handling
2. **Use caching wisely** - invalidate cache when data changes
3. **Implement loading states** for better UX
4. **Keep services focused** - each service should handle one domain
5. **Use TypeScript** for type safety and better DX

## API Service Architecture

The services use a layered architecture:

```
Component Layer
     ↓
Service Layer (orderService, productService, etc.)
     ↓
API Layer (api.ts)
     ↓
HTTP Layer (fetch)
```

This provides:
- **Separation of concerns**
- **Reusability**
- **Centralized error handling**
- **Consistent API interface**
- **Easy testing and mocking**
