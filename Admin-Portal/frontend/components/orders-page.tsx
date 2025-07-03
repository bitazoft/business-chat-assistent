"use client"

import { useState } from "react"
import { Search, Eye } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

const initialOrders = [
  {
    id: "#1234",
    customer: "John Doe",
    product: "iPhone 15 Pro",
    amount: "$999",
    date: "2024-01-15",
    status: "Completed",
  },
  {
    id: "#1235",
    customer: "Jane Smith",
    product: "Samsung Galaxy S24",
    amount: "$799",
    date: "2024-01-14",
    status: "Processing",
  },
  {
    id: "#1236",
    customer: "Mike Johnson",
    product: "MacBook Air M3",
    amount: "$1,199",
    date: "2024-01-14",
    status: "Pending",
  },
  {
    id: "#1237",
    customer: "Sarah Wilson",
    product: "AirPods Pro",
    amount: "$249",
    date: "2024-01-13",
    status: "Shipped",
  },
  {
    id: "#1238",
    customer: "David Brown",
    product: "iPad Air",
    amount: "$599",
    date: "2024-01-13",
    status: "Cancelled",
  },
  {
    id: "#1239",
    customer: "Lisa Davis",
    product: "iPhone 15 Pro",
    amount: "$999",
    date: "2024-01-12",
    status: "Completed",
  },
]

export function OrdersPage() {
  const [orders, setOrders] = useState(initialOrders)
  const [searchTerm, setSearchTerm] = useState("")
  const [statusFilter, setStatusFilter] = useState("All")

  const filteredOrders = orders.filter((order) => {
    const matchesSearch =
      order.customer.toLowerCase().includes(searchTerm.toLowerCase()) ||
      order.id.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesStatus = statusFilter === "All" || order.status === statusFilter
    return matchesSearch && matchesStatus
  })

  const getStatusColor = (status: string) => {
    switch (status) {
      case "Completed":
        return "bg-emerald-500/20 text-emerald-400"
      case "Processing":
        return "bg-blue-500/20 text-blue-400"
      case "Shipped":
        return "bg-purple-500/20 text-purple-400"
      case "Pending":
        return "bg-yellow-500/20 text-yellow-400"
      case "Cancelled":
        return "bg-red-500/20 text-red-400"
      default:
        return "bg-gray-500/20 text-gray-400"
    }
  }

  const statusOptions = ["All", "Completed", "Processing", "Shipped", "Pending", "Cancelled"]

  return (
    <div className="space-y-6 animate-in fade-in-50 duration-500">
      <div>
        <h2 className="text-3xl font-bold text-white mb-2">Orders</h2>
        <p className="text-gray-400">Track and manage customer orders</p>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4 animate-in slide-in-from-top duration-500">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
          <Input
            placeholder="Search orders or customers..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 bg-[#1a1a2e] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300"
          />
        </div>
        <div className="flex gap-2 flex-wrap">
          {statusOptions.map((status) => (
            <Button
              key={status}
              variant={statusFilter === status ? "default" : "outline"}
              size="sm"
              onClick={() => setStatusFilter(status)}
              className={
                statusFilter === status
                  ? "bg-gradient-to-r from-violet-500 to-purple-600 text-white"
                  : "border-gray-600 text-gray-300 hover:text-violet-400 hover:border-violet-400 hover:bg-violet-500/10"
              }
            >
              {status}
            </Button>
          ))}
        </div>
      </div>

      {/* Orders Table */}
      <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-bottom duration-700">
        <CardHeader>
          <CardTitle className="text-white">Orders ({filteredOrders.length})</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Order ID</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Customer</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Product</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Amount</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Date</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Status</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredOrders.map((order, index) => (
                  <tr
                    key={order.id}
                    className="border-b border-gray-800 hover:bg-gray-800/30 transition-all duration-200 animate-in slide-in-from-left"
                    style={{ animationDelay: `${index * 50}ms` }}
                  >
                    <td className="py-3 px-4 text-emerald-400 font-mono font-semibold">{order.id}</td>
                    <td className="py-3 px-4 text-white font-medium">{order.customer}</td>
                    <td className="py-3 px-4 text-gray-300">{order.product}</td>
                    <td className="py-3 px-4 text-violet-400 font-semibold">{order.amount}</td>
                    <td className="py-3 px-4 text-gray-300">{order.date}</td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-1 rounded-full text-xs ${getStatusColor(order.status)}`}>
                        {order.status}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <Button
                        size="sm"
                        variant="ghost"
                        className="text-violet-400 hover:text-emerald-400 hover:bg-gray-800/50"
                      >
                        <Eye className="w-4 h-4" />
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
