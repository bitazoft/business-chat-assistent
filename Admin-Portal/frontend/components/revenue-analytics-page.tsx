"use client"

import { TrendingUp, DollarSign, ShoppingCart, CalendarIcon, BarChart3, LineChart, Calendar as CalendarLucide } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Calendar } from "@/components/ui/calendar"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import analyticsService, { RevenueData, OrdersData } from "@/services/analyticService"
import { useEffect, useState } from "react"
import { format, subDays, subMonths, startOfMonth, endOfMonth, eachDayOfInterval, eachMonthOfInterval, subYears } from "date-fns"
import { getCurrentUser } from "@/lib/auth"
import { cn } from "@/lib/utils"
import {
  LineChart as RechartsLineChart,
  BarChart as RechartsBarChart,
  Area,
  AreaChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Bar,
  Line,
  PieChart,
  Pie,
  Cell
} from "recharts"

interface ChartDataPoint {
  date: string;
  revenue: number;
  orders: number;
  label: string;
}

interface MonthlyChartData {
  month: string;
  revenue: number;
  orders: number;
  label: string;
}

export function RevenueAnalyticsPage() {
  const [loading, setLoading] = useState(true)
  const [selectedDate, setSelectedDate] = useState<Date>(new Date())
  const [selectedYear, setSelectedYear] = useState<number>(new Date().getFullYear())
  const [selectedMonth, setSelectedMonth] = useState<number>(new Date().getMonth() + 1)
  const [viewMode, setViewMode] = useState<'daily' | 'monthly'>('daily')
  const [timeRange, setTimeRange] = useState<'7days' | '30days' | '12months'>('7days')
  
  const [dailyChartData, setDailyChartData] = useState<ChartDataPoint[]>([])
  const [monthlyChartData, setMonthlyChartData] = useState<MonthlyChartData[]>([])
  const [selectedDateRevenue, setSelectedDateRevenue] = useState<RevenueData | null>(null)
  const [selectedDateOrders, setSelectedDateOrders] = useState<OrdersData | null>(null)
  const [selectedMonthRevenue, setSelectedMonthRevenue] = useState<RevenueData | null>(null)
  const [selectedMonthOrders, setSelectedMonthOrders] = useState<OrdersData | null>(null)

  const user = getCurrentUser()

  const formatCurrency = (amount: number): string => {
    return `Rs. ${new Intl.NumberFormat('en-US', {
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount)}`;
  };

  const formatNumber = (num: number): string => {
    return new Intl.NumberFormat('en-US').format(num);
  };

  const generateDateRange = (days: number): Date[] => {
    const endDate = new Date()
    const startDate = subDays(endDate, days - 1)
    return eachDayOfInterval({ start: startDate, end: endDate })
  }

  const generateMonthRange = (months: number): Date[] => {
    const endDate = new Date()
    const startDate = subMonths(endDate, months - 1)
    return eachMonthOfInterval({ start: startOfMonth(startDate), end: endOfMonth(endDate) })
  }

  const fetchDailyData = async (days: number) => {
    if (!user?.sellerId) return

    const dates = generateDateRange(days)
    const promises = dates.map(async (date) => {
      const dateStr = format(date, 'yyyy-MM-dd')
      const [revenueData, ordersData] = await Promise.all([
        analyticsService.getRevenueByDate(user.sellerId!.toString(), dateStr),
        analyticsService.getOrdersByDate(user.sellerId!.toString(), dateStr)
      ])

      return {
        date: dateStr,
        revenue: revenueData?.totalRevenue || 0,
        orders: ordersData?.orderCount || 0,
        label: format(date, 'MMM dd')
      }
    })

    const results = await Promise.all(promises)
    setDailyChartData(results)
  }

  const fetchMonthlyData = async (months: number) => {
    if (!user?.sellerId) return

    const dates = generateMonthRange(months)
    const promises = dates.map(async (date) => {
      const year = date.getFullYear()
      const month = date.getMonth() + 1
      const [revenueData, ordersData] = await Promise.all([
        analyticsService.getRevenueByMonth(user.sellerId!.toString(), year, month),
        analyticsService.getOrdersByMonth(user.sellerId!.toString(), year, month)
      ])

      return {
        month: format(date, 'yyyy-MM'),
        revenue: revenueData?.totalRevenue || 0,
        orders: ordersData?.orderCount || 0,
        label: format(date, 'MMM yyyy')
      }
    })

    const results = await Promise.all(promises)
    setMonthlyChartData(results)
  }

  const fetchSelectedDateData = async () => {
    if (!user?.sellerId) return

    const dateStr = format(selectedDate, 'yyyy-MM-dd')
    const [revenueData, ordersData] = await Promise.all([
      analyticsService.getRevenueByDate(user.sellerId!.toString(), dateStr),
      analyticsService.getOrdersByDate(user.sellerId!.toString(), dateStr)
    ])

    setSelectedDateRevenue(revenueData)
    setSelectedDateOrders(ordersData)
  }

  const fetchSelectedMonthData = async () => {
    if (!user?.sellerId) return

    const [revenueData, ordersData] = await Promise.all([
      analyticsService.getRevenueByMonth(user.sellerId!.toString(), selectedYear, selectedMonth),
      analyticsService.getOrdersByMonth(user.sellerId!.toString(), selectedYear, selectedMonth)
    ])

    setSelectedMonthRevenue(revenueData)
    setSelectedMonthOrders(ordersData)
  }

  useEffect(() => {
    if (!user?.sellerId) {
      setLoading(false)
      return
    }

    const fetchData = async () => {
      setLoading(true)
      try {
        if (viewMode === 'daily') {
          await fetchSelectedDateData()
          const days = timeRange === '7days' ? 7 : 30
          await fetchDailyData(days)
        } else {
          await fetchSelectedMonthData()
          await fetchMonthlyData(12)
        }
      } catch (error) {
        console.error('Error fetching revenue analytics data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [selectedDate, selectedYear, selectedMonth, viewMode, timeRange, user?.sellerId])

  const currentChartData = viewMode === 'daily' ? dailyChartData : monthlyChartData
  const totalRevenue = currentChartData.reduce((sum, item) => sum + item.revenue, 0)
  const totalOrders = currentChartData.reduce((sum, item) => sum + item.orders, 0)
  const avgOrderValue = totalOrders > 0 ? totalRevenue / totalOrders : 0

  const COLORS = ['#8b5cf6', '#06d6a0', '#f72585', '#ffbe0b', '#fb8500']

  return (
    <div className="space-y-6 animate-in fade-in-50 duration-500">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-white mb-2">Revenue & Orders Analytics</h2>
          <p className="text-gray-400">Track your revenue and order trends with detailed insights</p>
        </div>
        
        <div className="flex items-center gap-4">
          <Tabs value={viewMode} onValueChange={(value) => setViewMode(value as 'daily' | 'monthly')}>
            <TabsList className="bg-[#1a1a2e] border-gray-600">
              <TabsTrigger value="daily" className="text-white data-[state=active]:bg-violet-600">Daily View</TabsTrigger>
              <TabsTrigger value="monthly" className="text-white data-[state=active]:bg-violet-600">Monthly View</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[
          {
            title: `Total Revenue (${timeRange === '7days' ? '7 Days' : timeRange === '30days' ? '30 Days' : '12 Months'})`,
            value: formatCurrency(totalRevenue),
            icon: DollarSign,
            color: "text-emerald-400",
            bgColor: "bg-emerald-500/10",
          },
          {
            title: `Total Orders (${timeRange === '7days' ? '7 Days' : timeRange === '30days' ? '30 Days' : '12 Months'})`,
            value: formatNumber(totalOrders),
            icon: ShoppingCart,
            color: "text-violet-400",
            bgColor: "bg-violet-500/10",
          },
          {
            title: "Average Order Value",
            value: formatCurrency(avgOrderValue),
            icon: TrendingUp,
            color: "text-blue-400",
            bgColor: "bg-blue-500/10",
          },
        ].map((metric, index) => {
          const Icon = metric.icon
          return (
            <Card
              key={index}
              className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 hover:border-violet-400/50 transition-all duration-300 hover:shadow-lg hover:shadow-violet-500/10"
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
                  <div className="text-2xl font-bold text-white">{metric.value}</div>
                )}
              </CardContent>
            </Card>
          )
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Revenue Chart */}
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-white flex items-center gap-2">
                  <DollarSign className="h-5 w-5 text-emerald-400" />
                  {viewMode === 'daily' ? 'Daily' : 'Monthly'} Revenue Trend
                </CardTitle>
                <p className="text-gray-400 text-sm">
                  Revenue performance over time
                </p>
              </div>
              {viewMode === 'daily' && (
                <Select value={timeRange} onValueChange={(value) => setTimeRange(value as any)}>
                  <SelectTrigger className="w-[140px] bg-[#0f0f23] border-gray-600 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#1a1a2e] border-gray-600">
                    <SelectItem value="7days" className="text-white hover:bg-gray-700">Last 7 Days</SelectItem>
                    <SelectItem value="30days" className="text-white hover:bg-gray-700">Last 30 Days</SelectItem>
                  </SelectContent>
                </Select>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex items-center justify-center h-80">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-400"></div>
              </div>
            ) : (
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={currentChartData}>
                    <defs>
                      <linearGradient id="revenueGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#10b981" stopOpacity={0.1}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="label" 
                      stroke="#9ca3af"
                      fontSize={12}
                    />
                    <YAxis 
                      stroke="#9ca3af"
                      fontSize={12}
                      tickFormatter={(value) => `$${value}`}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1a1a2e', 
                        border: '1px solid #374151',
                        borderRadius: '8px',
                        color: 'white'
                      }}
                      formatter={(value) => [formatCurrency(value as number), 'Revenue']}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="revenue" 
                      stroke="#10b981" 
                      strokeWidth={3}
                      fill="url(#revenueGradient)" 
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Orders Chart */}
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-white flex items-center gap-2">
                  <ShoppingCart className="h-5 w-5 text-violet-400" />
                  {viewMode === 'daily' ? 'Daily' : 'Monthly'} Orders Trend
                </CardTitle>
                <p className="text-gray-400 text-sm">
                  Order volume over time
                </p>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex items-center justify-center h-80">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-violet-400"></div>
              </div>
            ) : (
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <RechartsBarChart data={currentChartData}>
                    <defs>
                      <linearGradient id="ordersGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="label" 
                      stroke="#9ca3af"
                      fontSize={12}
                    />
                    <YAxis 
                      stroke="#9ca3af"
                      fontSize={12}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1a1a2e', 
                        border: '1px solid #374151',
                        borderRadius: '8px',
                        color: 'white'
                      }}
                      formatter={(value) => [value, 'Orders']}
                    />
                    <Bar 
                      dataKey="orders" 
                      fill="url(#ordersGradient)" 
                      radius={[4, 4, 0, 0]}
                    />
                  </RechartsBarChart>
                </ResponsiveContainer>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Date/Month Selector and Details */}
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <CalendarLucide className="h-5 w-5 text-violet-400" />
              {viewMode === 'daily' ? 'Date' : 'Month'} Details
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {viewMode === 'daily' ? (
              <>
                <Popover>
                  <PopoverTrigger asChild>
                    <Button
                      variant="outline"
                      className={cn(
                        "w-full justify-start text-left font-normal bg-[#0f0f23] border-gray-600 text-white hover:bg-gray-800/30",
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

                {selectedDateRevenue && selectedDateOrders && (
                  <div className="space-y-4">
                    <div className="bg-[#0f0f23] rounded-lg p-4">
                      <div className="text-2xl font-bold text-emerald-400 mb-1">
                        {formatCurrency(selectedDateRevenue.totalRevenue)}
                      </div>
                      <div className="text-sm text-gray-400">Revenue on {format(selectedDate, "MMM dd, yyyy")}</div>
                    </div>
                    
                    <div className="bg-[#0f0f23] rounded-lg p-4">
                      <div className="text-2xl font-bold text-violet-400 mb-1">
                        {formatNumber(selectedDateOrders.orderCount)}
                      </div>
                      <div className="text-sm text-gray-400">Orders on {format(selectedDate, "MMM dd, yyyy")}</div>
                    </div>

                    {selectedDateOrders.orderCount > 0 && (
                      <div className="bg-[#0f0f23] rounded-lg p-4">
                        <div className="text-lg font-bold text-blue-400 mb-1">
                          {formatCurrency(selectedDateRevenue.totalRevenue / selectedDateOrders.orderCount)}
                        </div>
                        <div className="text-sm text-gray-400">Avg. Order Value</div>
                      </div>
                    )}
                  </div>
                )}
              </>
            ) : (
              <>
                <div className="grid grid-cols-2 gap-2">
                  <Select value={selectedYear.toString()} onValueChange={(value) => setSelectedYear(parseInt(value))}>
                    <SelectTrigger className="bg-[#0f0f23] border-gray-600 text-white">
                      <SelectValue placeholder="Year" />
                    </SelectTrigger>
                    <SelectContent className="bg-[#1a1a2e] border-gray-600">
                      {Array.from({ length: 5 }, (_, i) => new Date().getFullYear() - i).map((year) => (
                        <SelectItem key={year} value={year.toString()} className="text-white hover:bg-gray-700">
                          {year}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>

                  <Select value={selectedMonth.toString()} onValueChange={(value) => setSelectedMonth(parseInt(value))}>
                    <SelectTrigger className="bg-[#0f0f23] border-gray-600 text-white">
                      <SelectValue placeholder="Month" />
                    </SelectTrigger>
                    <SelectContent className="bg-[#1a1a2e] border-gray-600">
                      {Array.from({ length: 12 }, (_, i) => i + 1).map((month) => (
                        <SelectItem key={month} value={month.toString()} className="text-white hover:bg-gray-700">
                          {format(new Date(2024, month - 1), 'MMMM')}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {selectedMonthRevenue && selectedMonthOrders && (
                  <div className="space-y-4">
                    <div className="bg-[#0f0f23] rounded-lg p-4">
                      <div className="text-2xl font-bold text-emerald-400 mb-1">
                        {formatCurrency(selectedMonthRevenue.totalRevenue)}
                      </div>
                      <div className="text-sm text-gray-400">
                        Revenue in {format(new Date(selectedYear, selectedMonth - 1), 'MMMM yyyy')}
                      </div>
                    </div>
                    
                    <div className="bg-[#0f0f23] rounded-lg p-4">
                      <div className="text-2xl font-bold text-violet-400 mb-1">
                        {formatNumber(selectedMonthOrders.orderCount)}
                      </div>
                      <div className="text-sm text-gray-400">
                        Orders in {format(new Date(selectedYear, selectedMonth - 1), 'MMMM yyyy')}
                      </div>
                    </div>

                    {selectedMonthOrders.orderCount > 0 && (
                      <div className="bg-[#0f0f23] rounded-lg p-4">
                        <div className="text-lg font-bold text-blue-400 mb-1">
                          {formatCurrency(selectedMonthRevenue.totalRevenue / selectedMonthOrders.orderCount)}
                        </div>
                        <div className="text-sm text-gray-400">Avg. Order Value</div>
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>

        {/* Revenue vs Orders Comparison Chart */}
        <Card className="lg:col-span-2 bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <LineChart className="h-5 w-5 text-blue-400" />
              Revenue vs Orders Correlation
            </CardTitle>
            <p className="text-gray-400 text-sm">
              Analyze the relationship between revenue and order volume
            </p>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex items-center justify-center h-80">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div>
              </div>
            ) : (
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <RechartsLineChart data={currentChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="label" 
                      stroke="#9ca3af"
                      fontSize={12}
                    />
                    <YAxis 
                      yAxisId="revenue"
                      orientation="left"
                      stroke="#10b981"
                      fontSize={12}
                      tickFormatter={(value) => `$${value}`}
                    />
                    <YAxis 
                      yAxisId="orders"
                      orientation="right"
                      stroke="#8b5cf6"
                      fontSize={12}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1a1a2e', 
                        border: '1px solid #374151',
                        borderRadius: '8px',
                        color: 'white'
                      }}
                      formatter={(value, name) => [
                        name === 'revenue' ? formatCurrency(value as number) : value,
                        name === 'revenue' ? 'Revenue' : 'Orders'
                      ]}
                    />
                    <Line 
                      yAxisId="revenue"
                      type="monotone" 
                      dataKey="revenue" 
                      stroke="#10b981" 
                      strokeWidth={3}
                      dot={{ fill: '#10b981', strokeWidth: 2, r: 4 }}
                      activeDot={{ r: 6, stroke: '#10b981', strokeWidth: 2 }}
                    />
                    <Line 
                      yAxisId="orders"
                      type="monotone" 
                      dataKey="orders" 
                      stroke="#8b5cf6" 
                      strokeWidth={3}
                      dot={{ fill: '#8b5cf6', strokeWidth: 2, r: 4 }}
                      activeDot={{ r: 6, stroke: '#8b5cf6', strokeWidth: 2 }}
                    />
                  </RechartsLineChart>
                </ResponsiveContainer>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Recent Orders Section */}
      {((viewMode === 'daily' && selectedDateOrders?.orders.length) || 
        (viewMode === 'monthly' && selectedMonthOrders?.orders.length)) && (
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700">
          <CardHeader>
            <CardTitle className="text-white">
              Recent Orders - {viewMode === 'daily' 
                ? format(selectedDate, "MMM dd, yyyy") 
                : format(new Date(selectedYear, selectedMonth - 1), 'MMMM yyyy')}
            </CardTitle>
            <p className="text-gray-400 text-sm">
              {viewMode === 'daily' 
                ? `${selectedDateOrders?.orders.length || 0} orders on this date`
                : `${selectedMonthOrders?.orders.length || 0} orders in this month`}
            </p>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {(viewMode === 'daily' ? selectedDateOrders?.orders : selectedMonthOrders?.orders)
                ?.slice(0, 10)
                .map((order: any, index: number) => (
                  <div
                    key={order.id || index}
                    className="flex items-center justify-between p-3 bg-[#0f0f23] rounded-lg hover:bg-gray-800/30 transition-colors duration-200"
                  >
                    <div className="flex-1">
                      <p className="text-white font-medium">Order #{order.id}</p>
                      <p className="text-gray-400 text-sm">
                        {format(new Date(order.created_at), "MMM dd, yyyy 'at' HH:mm")}
                      </p>
                      <p className="text-gray-400 text-xs">
                        {order.order_items?.length || 0} items • Status: {order.status}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="text-emerald-400 font-semibold">
                        {formatCurrency(order.total_amount)}
                      </p>
                    </div>
                  </div>
                ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
