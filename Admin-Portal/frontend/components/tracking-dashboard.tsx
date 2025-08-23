"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { initialOrders } from "@/components/orders-page"

export function TrackingDashboard() {
  const pendingOrders = initialOrders.filter(order => order.status === "Pending")

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-white">Tracking Dashboard</h2>
      <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700">
        <CardHeader>
          <CardTitle className="text-yellow-400">Pending Orders ({pendingOrders.length})</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-4">
            {pendingOrders.map((order) => (
              <li key={order.id} className="p-4 border border-yellow-500 rounded text-white">
                <div><strong>ID:</strong> {order.id}</div>
                <div><strong>Customer:</strong> {order.customer}</div>
                <div><strong>Product:</strong> {order.product}</div>
                <div><strong>Date:</strong> {order.date}</div>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}
