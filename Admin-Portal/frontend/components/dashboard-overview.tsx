"use client"

import { TrendingUp, Package, ShoppingCart, Users } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

const stats = [
  {
    title: "Total Products",
    value: "156",
    change: "+12%",
    icon: Package,
    color: "text-violet-400",
    bgColor: "bg-violet-500/10",
  },
  {
    title: "Active Orders",
    value: "89",
    change: "+8%",
    icon: ShoppingCart,
    color: "text-emerald-400",
    bgColor: "bg-emerald-500/10",
  },
  {
    title: "Active Users",
    value: "2,341",
    change: "+23%",
    icon: Users,
    color: "text-violet-400",
    bgColor: "bg-violet-500/10",
  },
  {
    title: "Trending Product",
    value: "iPhone 15",
    change: "+45%",
    icon: TrendingUp,
    color: "text-emerald-400",
    bgColor: "bg-emerald-500/10",
  },
]

export function DashboardOverview() {
  return (
    <div className="space-y-6 animate-in fade-in-50 duration-500">
      <div>
        <h2 className="text-3xl font-bold text-white mb-2">Dashboard Overview</h2>
        <p className="text-gray-400">Monitor your WhatsApp Business performance</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => {
          const Icon = stat.icon
          return (
            <Card
              key={index}
              className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 hover:border-violet-400/50 transition-all hover:shadow-lg hover:shadow-violet-500/10 animate-in slide-in-from-bottom-4 duration-700"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">{stat.title}</CardTitle>
                <div className={`p-2 rounded-lg ${stat.bgColor}`}>
                  <Icon className={`h-4 w-4 ${stat.color}`} />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-white">{stat.value}</div>
                <p className="text-xs text-emerald-400">{stat.change} from last month</p>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-left duration-700">
          <CardHeader>
            <CardTitle className="text-white">Recent Orders</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { id: "#1234", customer: "John Doe", amount: "$299", status: "Completed" },
                { id: "#1235", customer: "Jane Smith", amount: "$199", status: "Processing" },
                { id: "#1236", customer: "Mike Johnson", amount: "$399", status: "Pending" },
              ].map((order, index) => (
                <div
                  key={order.id}
                  className="flex items-center justify-between p-3 bg-[#0f0f23] rounded-lg hover:bg-gray-800/30 transition-colors duration-200 animate-in slide-in-from-bottom-2"
                  style={{ animationDelay: `${(index + 4) * 100}ms` }}
                >
                  <div>
                    <p className="text-white font-medium">{order.customer}</p>
                    <p className="text-gray-400 text-sm">{order.id}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-violet-400 font-semibold">{order.amount}</p>
                    <span
                      className={`text-xs px-2 py-1 rounded-full ${
                        order.status === "Completed"
                          ? "bg-emerald-500/20 text-emerald-400"
                          : order.status === "Processing"
                            ? "bg-yellow-500/20 text-yellow-400"
                            : "bg-red-500/20 text-red-400"
                      }`}
                    >
                      {order.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-right duration-700">
          <CardHeader>
            <CardTitle className="text-white">Top Products</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {[
                { name: "iPhone 15 Pro", sales: 45, revenue: "$13,500" },
                { name: "Samsung Galaxy S24", sales: 32, revenue: "$9,600" },
                { name: "MacBook Air M3", sales: 28, revenue: "$33,600" },
              ].map((product, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-[#0f0f23] rounded-lg hover:bg-gray-800/30 transition-colors duration-200 animate-in slide-in-from-bottom-2"
                  style={{ animationDelay: `${(index + 7) * 100}ms` }}
                >
                  <div>
                    <p className="text-white font-medium">{product.name}</p>
                    <p className="text-gray-400 text-sm">{product.sales} sales</p>
                  </div>
                  <p className="text-emerald-400 font-semibold">{product.revenue}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
