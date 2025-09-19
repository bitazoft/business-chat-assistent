"use client"

import { useState } from "react"
import { TrendingUp, Users, MessageSquare, ShoppingBag, DollarSign, Calendar, Activity, Globe } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"

export function PlatformInsightsPage() {
  const [timeRange, setTimeRange] = useState("30d")

  // Mock data - in real app, this would come from your analytics API
  const platformMetrics = {
    totalRevenue: "$245,670",
    revenueGrowth: "+23.5%",
    totalUsers: 1247,
    userGrowth: "+12.3%",
    totalOrders: 3456,
    orderGrowth: "+18.7%",
    totalMessages: 45678,
    messageGrowth: "+34.2%",
    averageOrderValue: "$71.05",
    conversionRate: "3.2%",
    activeUsers: 892,
    churnRate: "2.1%",
  }

  const revenueData = [
    { month: "Jan", revenue: 18500, users: 145 },
    { month: "Feb", revenue: 22300, users: 167 },
    { month: "Mar", revenue: 19800, users: 189 },
    { month: "Apr", revenue: 25600, users: 203 },
    { month: "May", revenue: 28900, users: 234 },
    { month: "Jun", revenue: 31200, users: 267 },
  ]

  const topPerformingUsers = [
    { name: "Fashion Boutique", revenue: "$12,450", orders: 45, growth: "+23%" },
    { name: "Food Delivery Co", revenue: "$8,920", orders: 128, growth: "+18%" },
    { name: "Sports Equipment", revenue: "$15,890", orders: 67, growth: "+31%" },
    { name: "Electronics Store", revenue: "$5,670", orders: 23, growth: "+12%" },
    { name: "Home Decor Studio", revenue: "$2,340", orders: 12, growth: "+8%" },
  ]

  const systemHealth = {
    uptime: "99.9%",
    responseTime: "245ms",
    errorRate: "0.02%",
    apiCalls: "1.2M",
    storageUsed: "67%",
    bandwidth: "2.3TB",
  }

  const geographicData = [
    { region: "North America", users: 456, percentage: 36.6 },
    { region: "Europe", users: 312, percentage: 25.0 },
    { region: "Asia Pacific", users: 289, percentage: 23.2 },
    { region: "Latin America", users: 134, percentage: 10.7 },
    { region: "Others", users: 56, percentage: 4.5 },
  ]

  return (
    <div className="space-y-6 animate-in fade-in-50 duration-500">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold text-white mb-2">Platform Insights</h2>
          <p className="text-gray-400">Comprehensive analytics and system performance metrics</p>
        </div>
        <div className="flex items-center space-x-3">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-32 bg-[#1a1a2e] border-gray-600 text-white">
              <Calendar className="w-4 h-4 mr-2" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-[#1a1a2e] border-gray-600">
              <SelectItem value="7d" className="text-white hover:bg-gray-800">
                Last 7 days
              </SelectItem>
              <SelectItem value="30d" className="text-white hover:bg-gray-800">
                Last 30 days
              </SelectItem>
              <SelectItem value="90d" className="text-white hover:bg-gray-800">
                Last 90 days
              </SelectItem>
              <SelectItem value="1y" className="text-white hover:bg-gray-800">
                Last year
              </SelectItem>
            </SelectContent>
          </Select>
          <Button
            variant="outline"
            className="border-gray-600 text-gray-300 hover:text-white hover:border-gray-500 bg-transparent"
          >
            Export Report
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 animate-in slide-in-from-top duration-500">
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 hover:border-violet-400/50 transition-all duration-300">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Total Revenue</p>
                <p className="text-2xl font-bold text-white">{platformMetrics.totalRevenue}</p>
                <p className="text-emerald-400 text-sm font-medium">{platformMetrics.revenueGrowth}</p>
              </div>
              <div className="p-3 rounded-lg bg-emerald-500/10">
                <DollarSign className="w-6 h-6 text-emerald-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 hover:border-violet-400/50 transition-all duration-300">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Total Users</p>
                <p className="text-2xl font-bold text-white">{platformMetrics.totalUsers.toLocaleString()}</p>
                <p className="text-violet-400 text-sm font-medium">{platformMetrics.userGrowth}</p>
              </div>
              <div className="p-3 rounded-lg bg-violet-500/10">
                <Users className="w-6 h-6 text-violet-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 hover:border-violet-400/50 transition-all duration-300">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Total Orders</p>
                <p className="text-2xl font-bold text-white">{platformMetrics.totalOrders.toLocaleString()}</p>
                <p className="text-emerald-400 text-sm font-medium">{platformMetrics.orderGrowth}</p>
              </div>
              <div className="p-3 rounded-lg bg-emerald-500/10">
                <ShoppingBag className="w-6 h-6 text-emerald-400" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 hover:border-violet-400/50 transition-all duration-300">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Messages Sent</p>
                <p className="text-2xl font-bold text-white">{platformMetrics.totalMessages.toLocaleString()}</p>
                <p className="text-violet-400 text-sm font-medium">{platformMetrics.messageGrowth}</p>
              </div>
              <div className="p-3 rounded-lg bg-violet-500/10">
                <MessageSquare className="w-6 h-6 text-violet-400" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Revenue Trend and Performance Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-left duration-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center">
              <TrendingUp className="w-5 h-5 mr-2 text-violet-400" />
              Revenue Trend
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {revenueData.map((data, index) => (
                <div key={data.month} className="flex items-center justify-between p-3 bg-[#0f0f23] rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="w-3 h-3 bg-violet-400 rounded-full"></div>
                    <span className="text-white font-medium">{data.month}</span>
                  </div>
                  <div className="text-right">
                    <p className="text-emerald-400 font-semibold">${data.revenue.toLocaleString()}</p>
                    <p className="text-gray-400 text-sm">{data.users} users</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-right duration-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center">
              <Activity className="w-5 h-5 mr-2 text-emerald-400" />
              Key Performance Indicators
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-[#0f0f23] rounded-lg">
                <p className="text-gray-400 text-sm">Avg Order Value</p>
                <p className="text-xl font-bold text-emerald-400">{platformMetrics.averageOrderValue}</p>
              </div>
              <div className="p-4 bg-[#0f0f23] rounded-lg">
                <p className="text-gray-400 text-sm">Conversion Rate</p>
                <p className="text-xl font-bold text-violet-400">{platformMetrics.conversionRate}</p>
              </div>
              <div className="p-4 bg-[#0f0f23] rounded-lg">
                <p className="text-gray-400 text-sm">Active Users</p>
                <p className="text-xl font-bold text-emerald-400">{platformMetrics.activeUsers}</p>
              </div>
              <div className="p-4 bg-[#0f0f23] rounded-lg">
                <p className="text-gray-400 text-sm">Churn Rate</p>
                <p className="text-xl font-bold text-yellow-400">{platformMetrics.churnRate}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Top Performing Users and Geographic Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-left duration-700 delay-200">
          <CardHeader>
            <CardTitle className="text-white flex items-center">
              <TrendingUp className="w-5 h-5 mr-2 text-emerald-400" />
              Top Performing Users
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {topPerformingUsers.map((user, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-[#0f0f23] rounded-lg hover:bg-gray-800/30 transition-colors duration-200"
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-gradient-to-r from-violet-500 to-purple-600 rounded-full flex items-center justify-center text-white font-semibold text-sm">
                      {index + 1}
                    </div>
                    <div>
                      <p className="text-white font-medium">{user.name}</p>
                      <p className="text-gray-400 text-sm">{user.orders} orders</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-emerald-400 font-semibold">{user.revenue}</p>
                    <p className="text-violet-400 text-sm">{user.growth}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-right duration-700 delay-200">
          <CardHeader>
            <CardTitle className="text-white flex items-center">
              <Globe className="w-5 h-5 mr-2 text-violet-400" />
              Geographic Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {geographicData.map((region, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-white font-medium">{region.region}</span>
                    <span className="text-gray-400">
                      {region.users} users ({region.percentage}%)
                    </span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-violet-500 to-purple-600 h-2 rounded-full transition-all duration-1000 ease-out"
                      style={{ width: `${region.percentage}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* System Health */}
      <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-bottom duration-700 delay-300">
        <CardHeader>
          <CardTitle className="text-white flex items-center">
            <Activity className="w-5 h-5 mr-2 text-emerald-400" />
            System Health & Performance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <div className="p-4 bg-[#0f0f23] rounded-lg text-center">
              <p className="text-gray-400 text-sm">Uptime</p>
              <p className="text-2xl font-bold text-emerald-400">{systemHealth.uptime}</p>
            </div>
            <div className="p-4 bg-[#0f0f23] rounded-lg text-center">
              <p className="text-gray-400 text-sm">Response Time</p>
              <p className="text-2xl font-bold text-violet-400">{systemHealth.responseTime}</p>
            </div>
            <div className="p-4 bg-[#0f0f23] rounded-lg text-center">
              <p className="text-gray-400 text-sm">Error Rate</p>
              <p className="text-2xl font-bold text-emerald-400">{systemHealth.errorRate}</p>
            </div>
            <div className="p-4 bg-[#0f0f23] rounded-lg text-center">
              <p className="text-gray-400 text-sm">API Calls</p>
              <p className="text-2xl font-bold text-violet-400">{systemHealth.apiCalls}</p>
            </div>
            <div className="p-4 bg-[#0f0f23] rounded-lg text-center">
              <p className="text-gray-400 text-sm">Storage Used</p>
              <p className="text-2xl font-bold text-yellow-400">{systemHealth.storageUsed}</p>
            </div>
            <div className="p-4 bg-[#0f0f23] rounded-lg text-center">
              <p className="text-gray-400 text-sm">Bandwidth</p>
              <p className="text-2xl font-bold text-emerald-400">{systemHealth.bandwidth}</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
