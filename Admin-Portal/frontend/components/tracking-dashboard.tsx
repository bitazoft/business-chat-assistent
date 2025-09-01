"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { fetchOrders, Order } from "@/services/orderService"
import { getCurrentUser } from "@/lib/auth"

export function TrackingDashboard() {
  const [orders, setOrders] = useState<Order[]>([])
  const [loading, setLoading] = useState(true)
  const user = getCurrentUser()

  useEffect(() => {
    loadOrders()
  }, [])

  const loadOrders = async () => {
    if (!user?.sellerId) {
      setLoading(false)
      return
    }

    try {
      const ordersData = await fetchOrders(user.sellerId.toString())
      setOrders(ordersData)
    } catch (error) {
      console.error("Failed to load orders:", error)
    } finally {
      setLoading(false)
    }
  }

  const pendingOrders = orders.filter((order: Order) => order.status === "Pending")

  if (loading) {
    return (
      <div className="space-y-6">
        <h2 className="text-2xl font-bold text-white">Tracking Dashboard</h2>
        <div className="text-center text-gray-400">Loading...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-white">Tracking Dashboard</h2>
      <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700">
        <CardHeader>
          <CardTitle className="text-yellow-400">Pending Orders ({pendingOrders.length})</CardTitle>
        </CardHeader>
        <CardContent>
          {pendingOrders.length === 0 ? (
            <div className="text-center text-gray-400 py-4">No pending orders</div>
          ) : (
            <ul className="space-y-4">
              {pendingOrders.map((order) => (
                <li key={order.id} className="p-4 border border-yellow-500 rounded text-white">
                  <div><strong>ID:</strong> {order.id}</div>
                  <div><strong>Customer:</strong> {order.customer}</div>
                  <div><strong>Products:</strong> {order.orderItems.map(item => item.name).join(', ') || 'No products'}</div>
                  <div><strong>Date:</strong> {order.date}</div>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
