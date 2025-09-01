"use client"

import { TrendingUp, Users, MessageSquare, ShoppingBag, CalendarIcon } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Calendar } from "@/components/ui/calendar"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import analyticsSearvice, { OverviewData,PopularProductsData,DailyEngagementData, CustomerInsightsData } from "@/services/analyticService"
import { use, useEffect, useState } from "react"
import { format } from "date-fns"
import { getCurrentUser } from "@/lib/auth"
import { cn } from "@/lib/utils"

export function AnalyticsPage() {

  const[overviewData, setOverviewData] = useState<OverviewData | null>(null)
  const [loading, setLoading] = useState(true)
  const [popularProductsData, setPopularProductsData] = useState<PopularProductsData[]>([])
  const [dailyEngagementData, setDailyEngagementData] = useState<DailyEngagementData[]>([])
  const [customerInsightsData, setCustomerInsightsData] = useState<CustomerInsightsData[]>([])
  const [totalCustomers, setTotalCustomers] = useState<number>(0)
  const [avgResponseTime, setAvgResponseTime] = useState<number>(0)
  const [engagedCustomers, setEngagedCustomers] = useState<number>(0)
  const [selectedDate, setSelectedDate] = useState<Date>(new Date())
  const user = getCurrentUser()


  const calculatePercentageChange = (current: number, previous: number): string => {
    if (previous === 0) return current > 0 ? "+100%" : "0%";
    const change = ((current - previous) / previous) * 100;
    return `${change >= 0 ? '+' : ''}${change.toFixed(1)}%`;
  };

  const formatCurrency = (amount: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const formatNumber = (num: number): string => {
    return new Intl.NumberFormat('en-US').format(num);
  };

  const metrics = [
    {
      title: "Total Revenue",
      value: overviewData ? formatCurrency(overviewData.totalProfit) : "$0",
      change: overviewData ? calculatePercentageChange(overviewData.totalProfitThisMonth, overviewData.totalProfitLastMonth) : "0%",
      icon: TrendingUp,
      color: "text-violet-400",
      bgColor: "bg-violet-500/10",
    },
    {
      title: "Active Customers",
      value: overviewData ? formatNumber(overviewData.totalUsers) : "0",
      change: overviewData ? calculatePercentageChange(overviewData.totalUsersThisMonth, overviewData.totalUsersLastMonth) : "0%",
      icon: Users,
      color: "text-emerald-400",
      bgColor: "bg-emerald-500/10",
    },
    {
      title: "Total Orders",
      value: overviewData ? formatNumber(overviewData.totalOrders) : "0",
      change: overviewData ? calculatePercentageChange(overviewData.activeOrders, overviewData.totalOrders - overviewData.activeOrders) : "0%",
      icon: MessageSquare,
      color: "text-violet-400",
      bgColor: "bg-violet-500/10",
    },
    {
      title: "Total Products",
      value: overviewData ? formatNumber(overviewData.totalProducts) : "0",
      icon: ShoppingBag,
      color: "text-emerald-400",
      bgColor: "bg-emerald-500/10",
    },
  ]
  useEffect(() => {
    if (!user?.sellerId) {
      setLoading(false)
      return
    }

    const fetchData = async () => {
      setLoading(true);
      try {
        const [overview, dailyEngagement, popularProducts, avgTime, engagedCustomers] = await Promise.all([
          analyticsSearvice.fetchOverview(user.sellerId!.toString()),
          analyticsSearvice.getDailyEngagementByDate(user.sellerId!.toString(), format(selectedDate, "yyyy-MM-dd")),
          analyticsSearvice.getPopularProducts(user.sellerId!.toString()),
          analyticsSearvice.getAvgResponseTime(user.sellerId!.toString()),
          analyticsSearvice.getCustomerByMessageCount(user.sellerId!.toString(), 10)
        ]);

        if (overview) setOverviewData(overview);
        if (dailyEngagement) setDailyEngagementData(dailyEngagement);
        if (popularProducts) setPopularProductsData(popularProducts);
        if (avgTime) setAvgResponseTime(avgTime);
        if (engagedCustomers) setEngagedCustomers(engagedCustomers);
      } catch (error) {
        console.error('Error fetching analytics data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [selectedDate, user?.sellerId]);


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
                {loading ? (
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-violet-400"></div>
                    <div className="text-gray-400">Loading...</div>
                  </div>
                ) : (
                  <>
                    <div className="text-2xl font-bold text-white">{metric.value}</div>
                    {metric.change && metric.change !== 'undefined' && metric.change !== '0%' && (
                      <p className={`text-xs ${metric.change.startsWith('-') ? 'text-red-400' : 'text-emerald-400'}`}>
                        {metric.change} from last month
                      </p>
                    )}
                  </>
                )}
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
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-violet-400"></div>
              </div>
            ) : (
              <div className="space-y-4">
                {popularProductsData.length > 0 ? (
                  popularProductsData.map((product, index) => (
                    <div
                      key={product.id || index}
                      className="flex items-center justify-between p-3 bg-[#0f0f23] rounded-lg hover:bg-gray-800/30 transition-colors duration-200 animate-in slide-in-from-bottom-2"
                      style={{ animationDelay: `${(index + 4) * 100}ms` }}
                    >
                      <div className="flex-1">
                        <p className="text-white font-medium">{product.name}</p>
                        <p className="text-gray-400 text-sm">{product.queries} queries</p>
                      </div>
                      <div className="text-right">
                        <p className="text-violet-400 font-semibold">
                          {product.queries > 0 ? Math.round((product.queries / Math.max(...popularProductsData.map(p => p.queries), 1)) * 100) : 0}%
                        </p>
                        <p className="text-gray-400 text-xs">engagement</p>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8">
                    <div className="text-gray-400 mb-4">
                      <ShoppingBag className="h-12 w-12 mx-auto mb-2 opacity-50" />
                      <p>No popular products data available</p>
                      <p className="text-sm mt-1">Start engaging with customers to see popular products</p>
                    </div>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Engagement Trends */}
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-right duration-700">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-white">Daily Engagement</CardTitle>
                <p className="text-gray-400 text-sm">Message volume throughout the day</p>
              </div>
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    className={cn(
                      "w-[200px] justify-start text-left font-normal bg-[#0f0f23] border-gray-600 text-white hover:bg-gray-800/30",
                      !selectedDate && "text-muted-foreground"
                    )}
                  >
                    <CalendarIcon className="mr-2 h-4 w-4 text-violet-400" />
                    {selectedDate ? format(selectedDate, "PPP") : <span>Pick a date</span>}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0 bg-[#1a1a2e] border-gray-600">
                  <Calendar
                    mode="single"
                    selected={selectedDate}
                    onSelect={(date) => date && setSelectedDate(date)}
                    initialFocus
                    className="text-white"
                  />
                </PopoverContent>
              </Popover>
            </div>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-violet-400"></div>
              </div>
            ) : (
              <div className="space-y-4">
                {dailyEngagementData.length > 0 ? (
                  dailyEngagementData.map((trend, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between animate-in slide-in-from-right"
                      style={{ animationDelay: `${(index + 9) * 100}ms` }}
                    >
                      <span className="text-gray-400 text-sm w-16">{trend.period || `${index + 9}:00`}</span>
                      <div className="flex-1 mx-4">
                        <div className="bg-[#0f0f23] rounded-full h-2 relative overflow-hidden">
                          <div
                            className="bg-gradient-to-r from-violet-500 to-emerald-500 h-full rounded-full transition-all duration-1000 ease-out"
                            style={{
                              width: `${Math.min((trend.messageCount || 0) / Math.max(...dailyEngagementData.map(d => d.messageCount || 0), 1) * 100, 100)}%`,
                              animationDelay: `${(index + 9) * 100 + 500}ms`,
                            }}
                          ></div>
                        </div>
                      </div>
                      <span className="text-violet-400 font-semibold text-sm w-12 text-right">{trend.messageCount || 0}</span>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-4">
                    <p className="text-gray-400">No data available for {format(selectedDate, "PPP")}</p>
                  </div>
                )}
              </div>
            )}
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
              { 
                value: overviewData ? `${Math.round((engagedCustomers / Math.max(overviewData.totalUsers, 1)) * 100)}%` : "0%", 
                label: "Returning Users", 
                color: "text-violet-400" 
              },
              { 
                value: avgResponseTime ? `${avgResponseTime/1000}` : "0", 
                label: "Avg. Response Time (s)", 
                color: "text-emerald-400" 
              },
              { 
                value: overviewData ? `${Math.round((overviewData.activeOrders / Math.max(overviewData.totalOrders, 1)) * 100)}%` : "0%", 
                label: "Active Orders Rate", 
                color: "text-violet-400" 
              },
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
