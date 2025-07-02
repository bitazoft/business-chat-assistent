"use client"

import { TrendingUp, Users, MessageSquare, ShoppingBag } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export function AnalyticsPage() {
  const metrics = [
    {
      title: "Total Revenue",
      value: "$45,231",
      change: "+20.1%",
      icon: TrendingUp,
      color: "text-violet-400",
      bgColor: "bg-violet-500/10",
    },
    {
      title: "Active Customers",
      value: "2,341",
      change: "+15.3%",
      icon: Users,
      color: "text-emerald-400",
      bgColor: "bg-emerald-500/10",
    },
    {
      title: "Messages Sent",
      value: "12,543",
      change: "+8.2%",
      icon: MessageSquare,
      color: "text-violet-400",
      bgColor: "bg-violet-500/10",
    },
    {
      title: "Products Sold",
      value: "1,234",
      change: "+12.5%",
      icon: ShoppingBag,
      color: "text-emerald-400",
      bgColor: "bg-emerald-500/10",
    },
  ]

  const popularProducts = [
    { name: "iPhone 15 Pro", queries: 245, engagement: "85%" },
    { name: "Samsung Galaxy S24", queries: 189, engagement: "78%" },
    { name: "MacBook Air M3", queries: 156, engagement: "92%" },
    { name: "AirPods Pro", queries: 134, engagement: "71%" },
    { name: "iPad Air", queries: 98, engagement: "66%" },
  ]

  const engagementTrends = [
    { time: "9 AM", messages: 45 },
    { time: "12 PM", messages: 78 },
    { time: "3 PM", messages: 92 },
    { time: "6 PM", messages: 156 },
    { time: "9 PM", messages: 134 },
  ]

  return (
    <div className="space-y-6 animate-in fade-in-50 duration-500">
      <div>
        <h2 className="text-3xl font-bold text-white mb-2">Analytics</h2>
        <p className="text-gray-400">Track your business performance and customer engagement</p>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metrics.map((metric, index) => {
          const Icon = metric.icon
          return (
            <Card
              key={index}
              className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 hover:border-violet-400/50 transition-all duration-300 hover:shadow-lg hover:shadow-violet-500/10 animate-in slide-in-from-bottom-4 duration-700"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">{metric.title}</CardTitle>
                <div className={`p-2 rounded-lg ${metric.bgColor}`}>
                  <Icon className={`h-4 w-4 ${metric.color}`} />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-white">{metric.value}</div>
                <p className="text-xs text-emerald-400">{metric.change} from last month</p>
              </CardContent>
            </Card>
          )
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Popular Products */}
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-left duration-700">
          <CardHeader>
            <CardTitle className="text-white">Popular Products</CardTitle>
            <p className="text-gray-400 text-sm">Most queried products by customers</p>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {popularProducts.map((product, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-[#0f0f23] rounded-lg hover:bg-gray-800/30 transition-colors duration-200 animate-in slide-in-from-bottom-2"
                  style={{ animationDelay: `${(index + 4) * 100}ms` }}
                >
                  <div className="flex-1">
                    <p className="text-white font-medium">{product.name}</p>
                    <p className="text-gray-400 text-sm">{product.queries} queries</p>
                  </div>
                  <div className="text-right">
                    <p className="text-violet-400 font-semibold">{product.engagement}</p>
                    <p className="text-gray-400 text-xs">engagement</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Engagement Trends */}
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-right duration-700">
          <CardHeader>
            <CardTitle className="text-white">Daily Engagement</CardTitle>
            <p className="text-gray-400 text-sm">Message volume throughout the day</p>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {engagementTrends.map((trend, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between animate-in slide-in-from-right"
                  style={{ animationDelay: `${(index + 9) * 100}ms` }}
                >
                  <span className="text-gray-400 text-sm w-16">{trend.time}</span>
                  <div className="flex-1 mx-4">
                    <div className="bg-[#0f0f23] rounded-full h-2 relative overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-violet-500 to-emerald-500 h-full rounded-full transition-all duration-1000 ease-out"
                        style={{
                          width: `${(trend.messages / 156) * 100}%`,
                          animationDelay: `${(index + 9) * 100 + 500}ms`,
                        }}
                      ></div>
                    </div>
                  </div>
                  <span className="text-violet-400 font-semibold text-sm w-12 text-right">{trend.messages}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Customer Insights */}
      <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-bottom duration-700 delay-300">
        <CardHeader>
          <CardTitle className="text-white">Customer Insights</CardTitle>
          <p className="text-gray-400 text-sm">Key metrics about your customer base</p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { value: "68%", label: "Returning Customers", color: "text-violet-400" },
              { value: "4.2", label: "Avg. Response Time (min)", color: "text-emerald-400" },
              { value: "92%", label: "Customer Satisfaction", color: "text-violet-400" },
            ].map((insight, index) => (
              <div
                key={index}
                className="text-center p-4 bg-[#0f0f23] rounded-lg hover:bg-gray-800/30 transition-colors duration-200 animate-in zoom-in-50"
                style={{ animationDelay: `${(index + 14) * 150}ms` }}
              >
                <div className={`text-3xl font-bold ${insight.color} mb-2`}>{insight.value}</div>
                <p className="text-gray-400">{insight.label}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
