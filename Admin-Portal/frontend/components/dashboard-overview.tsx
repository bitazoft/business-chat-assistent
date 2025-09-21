import { useEffect, useState } from "react"
import { TrendingUp, Package, ShoppingCart, Users } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import dashboardService, { OverviewData, TopProductsData, RecentOrdersData } from "@/services/dashboardService"
import { getCurrentUser } from "@/lib/auth"

export function DashboardOverview() {
  const [overviewData, setOverviewData] = useState<OverviewData | null>(null)
  const [topProducts, setTopProducts] = useState<TopProductsData[]>([])
  const [recentOrders, setRecentOrders] = useState<RecentOrdersData[]>([])
  const [loading, setLoading] = useState(true)
  const user = getCurrentUser()

  useEffect(() => {
    if (!user?.sellerId) {
      setLoading(false)
      return
    }
    const loadDashboardData = async () => {
      setLoading(true)
      try {
        const [overview, products, orders] = await Promise.all([
          dashboardService.fetchOverview(user.sellerId!.toString()),
          dashboardService.fetchTopProducts(user.sellerId!.toString()),
          dashboardService.fetchRecentOrders(user.sellerId!.toString())
        ])
        
        if (overview) setOverviewData(overview)
        if (products && Array.isArray(products)) setTopProducts(products)
        if (orders && Array.isArray(orders)) setRecentOrders(orders)
      } catch (error) {
        console.error("Error loading dashboard data:", error)
      } finally {
        setLoading(false)
      }
    }

    loadDashboardData()
  }, [])

  // Calculate percentage changes
  const calculateChange = (current: number, previous: number) => {
    if (typeof current !== 'number' || typeof previous !== 'number') return "+0%"
    if (previous === 0) return "+100%"
    const change = ((current - previous) / previous) * 100
    return `${change >= 0 ? '+' : ''}${change.toFixed(1)}%`
  }

  const stats = overviewData && typeof overviewData === 'object' ? [
    {
      title: "Total Products",
      value: (overviewData.totalProducts || 0).toString(),
      // change: "+12%", // You might want to calculate this based on historical data
      icon: Package,
      color: "text-violet-400",
      bgColor: "bg-violet-500/10",
    },
    {
      title: "Active Orders",
      value: (overviewData.activeOrders || 0).toString(),
      // change: "+8%", // You might want to calculate this based on historical data
      icon: ShoppingCart,
      color: "text-emerald-400",
      bgColor: "bg-emerald-500/10",
    },
    {
      title: "Total Users This Month",
      value: (overviewData.totalUsersThisMonth || 0).toLocaleString(),
      change: calculateChange(overviewData.totalUsersThisMonth || 0, overviewData.totalUsersLastMonth || 0),
      icon: Users,
      color: "text-violet-400",
      bgColor: "bg-violet-500/10",
    },
    {
      title: "Total Profit This Month",
      value: `Rs.${(overviewData.totalProfitThisMonth || 0).toLocaleString()}`,
      change: calculateChange(overviewData.totalProfitThisMonth || 0, overviewData.totalProfitLastMonth || 0),
      icon: TrendingUp,
      color: "text-emerald-400",
      bgColor: "bg-emerald-500/10",
    },
    {
      title: "Total Users Today",
      value: (overviewData.totalUsersToday || 0).toLocaleString(),
      icon: Users,
      color: "text-blue-400",
      bgColor: "bg-blue-500/10",
    },
    {
      title: "Total Profit Today",
      value: `Rs.${(overviewData.totalProfitToday || 0).toLocaleString()}`,
      icon: TrendingUp,
      color: "text-green-400",
      bgColor: "bg-green-500/10",
    },
  ] : []

  if (loading) {
    return (
      <div className="space-y-6 animate-in fade-in-50 duration-500">
        <div>
          <h2 className="text-3xl font-bold text-white mb-2">Dashboard Overview</h2>
          <p className="text-gray-400">Monitor your WhatsApp Business performance</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[...Array(6)].map((_, index) => (
            <Card
              key={`loading-card-${index}`}
              className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-pulse"
            >
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <div className="h-4 bg-gray-600 rounded w-20"></div>
                <div className="h-8 w-8 bg-gray-600 rounded-lg"></div>
              </CardHeader>
              <CardContent>
                <div className="h-8 bg-gray-600 rounded w-16 mb-2"></div>
                <div className="h-3 bg-gray-600 rounded w-24"></div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    )
  }
  return (
    <div className="space-y-6 animate-in fade-in-50 duration-500">
      <div>
        <h2 className="text-3xl font-bold text-white mb-2">Dashboard Overview</h2>
        <p className="text-gray-400">Monitor your WhatsApp Business performance</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-6">
        {stats.map((stat, index) => {
          const Icon = stat.icon
          return (
            <Card
              key={index}
              className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 hover:border-violet-400/50 transition-all duration-300 hover:shadow-lg hover:shadow-violet-500/10 animate-in slide-in-from-bottom-4 duration-700"
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
                {stat.change && stat.change !== 'undefined' && stat.change !== '0%' && (
                      <p className={`text-xs ${stat.change.startsWith('-') ? 'text-red-400' : 'text-emerald-400'}`}>
                        {stat.change} from last month
                      </p>
                    )}
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
              {recentOrders.length > 0 ? recentOrders.slice(0, 3).map((order, index) => (
                <div
                  key={order.orderId}
                  className="flex items-center justify-between p-3 bg-[#0f0f23] rounded-lg hover:bg-gray-800/30 transition-colors duration-200 animate-in slide-in-from-bottom-2"
                  style={{ animationDelay: `${(index + 4) * 100}ms` }}
                >
                  <div>
                    <p className="text-white font-medium">{order.customerName}</p>
                    <p className="text-gray-400 text-sm">#{order.orderId}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-violet-400 font-semibold">Rs.{(order.totalAmount || 0).toFixed(2)}</p>
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
              )) : (
                <div className="text-center text-gray-400 py-8">
                  <p>No recent orders available</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-right duration-700">
          <CardHeader>
            <CardTitle className="text-white">Top Products</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {topProducts.length > 0 ? topProducts.slice(0, 3).map((product, index) => (
                <div
                  key={product.productId}
                  className="flex items-center justify-between p-3 bg-[#0f0f23] rounded-lg hover:bg-gray-800/30 transition-colors duration-200 animate-in slide-in-from-bottom-2"
                  style={{ animationDelay: `${(index + 7) * 100}ms` }}
                >
                  <div>
                    <p className="text-white font-medium">{product.productName}</p>
                    <p className="text-gray-400 text-sm">{product.totalSales} sales</p>
                  </div>
                  <p className="text-emerald-400 font-semibold">Rs.{(product.totalQuantity || 0).toFixed(2)}</p>
                </div>
              )) : (
                <div className="text-center text-gray-400 py-8">
                  <p>No product data available</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
