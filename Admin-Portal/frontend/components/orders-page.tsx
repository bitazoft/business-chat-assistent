"use client"

import { useState } from "react"
import { Search, Eye, ArrowLeft, Package, X, Save, Edit } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface OrderItem {
  id: string
  name: string
  quantity: number
  price: number
  total: number
}

interface Order {
  id: string
  customer: string
  phone: string
  address: string
  netValue: number
  date: string
  status: string
  orderItems: OrderItem[]
  notes?: string
}

const initialOrders: Order[] = [
  {
    id: "#1234",
    customer: "John Doe",
    phone: "+1 (555) 123-4567",
    address: "123 Main St, New York, NY 10001",
    netValue: 1247,
    date: "2024-01-15",
    status: "Completed",
    orderItems: [
      { id: "1", name: "iPhone 15 Pro", quantity: 1, price: 999, total: 999 },
      { id: "2", name: "Phone Case", quantity: 1, price: 29, total: 29 },
      { id: "3", name: "Screen Protector", quantity: 2, price: 15, total: 30 },
    ],
    notes: "Customer requested express delivery",
  },
  {
    id: "#1235",
    customer: "Jane Smith",
    phone: "+1 (555) 234-5678",
    address: "456 Oak Ave, Los Angeles, CA 90210",
    netValue: 829,
    date: "2024-01-14",
    status: "Processing",
    orderItems: [
      { id: "1", name: "Samsung Galaxy S24", quantity: 1, price: 799, total: 799 },
      { id: "2", name: "Wireless Charger", quantity: 1, price: 30, total: 30 },
    ],
    notes: "Gift wrapping requested",
  },
  {
    id: "#1236",
    customer: "Mike Johnson",
    phone: "+1 (555) 345-6789",
    address: "789 Pine St, Chicago, IL 60601",
    netValue: 1299,
    date: "2024-01-14",
    status: "Pending",
    orderItems: [
      { id: "1", name: "MacBook Air M3", quantity: 1, price: 1199, total: 1199 },
      { id: "2", name: "USB-C Hub", quantity: 1, price: 49, total: 49 },
      { id: "3", name: "Laptop Sleeve", quantity: 1, price: 25, total: 25 },
      { id: "4", name: "Wireless Mouse", quantity: 1, price: 35, total: 35 },
    ],
  },
  {
    id: "#1237",
    customer: "Sarah Wilson",
    phone: "+1 (555) 456-7890",
    address: "321 Elm St, Miami, FL 33101",
    netValue: 279,
    date: "2024-01-13",
    status: "Shipped",
    orderItems: [
      { id: "1", name: "AirPods Pro", quantity: 1, price: 249, total: 249 },
      { id: "2", name: "Cleaning Kit", quantity: 1, price: 15, total: 15 },
      { id: "3", name: "Carrying Case", quantity: 1, price: 20, total: 20 },
    ],
  },
  {
    id: "#1238",
    customer: "David Brown",
    phone: "+1 (555) 567-8901",
    address: "654 Maple Dr, Seattle, WA 98101",
    netValue: 649,
    date: "2024-01-13",
    status: "Cancelled",
    orderItems: [
      { id: "1", name: "iPad Air", quantity: 1, price: 599, total: 599 },
      { id: "2", name: "Apple Pencil", quantity: 1, price: 79, total: 79 },
    ],
    notes: "Customer cancelled due to budget constraints",
  },
  {
    id: "#1239",
    customer: "Lisa Davis",
    phone: "+1 (555) 678-9012",
    address: "987 Cedar Ln, Boston, MA 02101",
    netValue: 1049,
    date: "2024-01-12",
    status: "Completed",
    orderItems: [
      { id: "1", name: "iPhone 15 Pro", quantity: 1, price: 999, total: 999 },
      { id: "2", name: "MagSafe Charger", quantity: 1, price: 39, total: 39 },
      { id: "3", name: "Lightning Cable", quantity: 1, price: 19, total: 19 },
    ],
  },
]

export function OrdersPage() {
  const [orders, setOrders] = useState(initialOrders)
  const [searchTerm, setSearchTerm] = useState("")
  const [statusFilter, setStatusFilter] = useState("All")
  const [selectedOrder, setSelectedOrder] = useState<Order | null>(null)
  const [showOrderDetails, setShowOrderDetails] = useState(false)
  const [showEditModal, setShowEditModal] = useState(false)
  const [editingOrder, setEditingOrder] = useState<Order | null>(null)

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

  const handleOrderClick = (order: Order) => {
    setSelectedOrder(order)
    setShowOrderDetails(true)
  }

  const handleBackToOrders = () => {
    setShowOrderDetails(false)
    setSelectedOrder(null)
  }

  const handleEditOrder = (order: Order) => {
    setEditingOrder({ ...order })
    setShowEditModal(true)
  }

  const handleSaveOrder = () => {
    if (editingOrder) {
      // Calculate new net value based on order items
      const subtotal = editingOrder.orderItems.reduce((sum, item) => sum + item.total, 0)
      const tax = subtotal * 0.08
      const shipping = 15
      const newNetValue = subtotal + tax + shipping

      const updatedOrder = {
        ...editingOrder,
        netValue: Math.round(newNetValue),
      }

      setOrders(orders.map((order) => (order.id === updatedOrder.id ? updatedOrder : order)))

      // Update selected order if it's the same one being edited
      if (selectedOrder && selectedOrder.id === updatedOrder.id) {
        setSelectedOrder(updatedOrder)
      }

      setShowEditModal(false)
      setEditingOrder(null)
    }
  }

  const handleCloseEditModal = () => {
    setShowEditModal(false)
    setEditingOrder(null)
  }

  const updateEditingOrderField = (field: keyof Order, value: any) => {
    if (editingOrder) {
      setEditingOrder({
        ...editingOrder,
        [field]: value,
      })
    }
  }

  const updateOrderItem = (itemId: string, field: keyof OrderItem, value: any) => {
    if (editingOrder) {
      const updatedItems = editingOrder.orderItems.map((item) => {
        if (item.id === itemId) {
          const updatedItem = { ...item, [field]: value }
          // Recalculate total when quantity or price changes
          if (field === "quantity" || field === "price") {
            updatedItem.total = updatedItem.quantity * updatedItem.price
          }
          return updatedItem
        }
        return item
      })

      setEditingOrder({
        ...editingOrder,
        orderItems: updatedItems,
      })
    }
  }

  const calculateSubtotal = (items: OrderItem[]) => {
    return items.reduce((sum, item) => sum + item.total, 0)
  }

  const calculateTax = (subtotal: number) => {
    return subtotal * 0.08 // 8% tax
  }

  const calculateShipping = () => {
    return 15 // Fixed shipping cost
  }

  // Show order details view
  if (showOrderDetails && selectedOrder) {
    return (
      <div className="space-y-6 animate-in fade-in-50 duration-500">
        {/* Header */}
        <div className="animate-in slide-in-from-top duration-500">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Button
                variant="ghost"
                onClick={handleBackToOrders}
                className="text-emerald-400 hover:text-white hover:bg-gray-800/50 mr-4 transition-all duration-200 hover:scale-105"
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to orders
              </Button>
              <div>
                <h2 className="text-3xl font-bold text-white">{selectedOrder.id}</h2>
                <p className="text-gray-400">{selectedOrder.customer}</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <div className="flex items-center mr-4">
                <span
                  className={`px-3 py-1 rounded-full text-sm ${getStatusColor(selectedOrder.status)} animate-pulse`}
                >
                  {selectedOrder.status}
                </span>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleEditOrder(selectedOrder)}
                className="border-gray-600 text-gray-300 hover:text-violet-400 hover:border-violet-400 bg-transparent transition-all duration-200 hover:scale-105"
              >
                <Edit className="w-4 h-4 mr-2" />
                Edit
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="border-gray-600 text-gray-300 hover:text-violet-400 hover:border-violet-400 bg-transparent transition-all duration-200 hover:scale-105"
              >
                Export
              </Button>
              <Button
                size="sm"
                className="bg-emerald-500 hover:bg-emerald-600 text-white transition-all duration-200 hover:scale-105 hover:shadow-lg hover:shadow-emerald-500/25"
              >
                Complete Order
              </Button>
            </div>
          </div>
        </div>

        {/* Order Details Section */}
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 hover:border-violet-400/50 transition-all duration-300 hover:shadow-lg hover:shadow-violet-500/10 animate-in slide-in-from-left duration-500 delay-100">
          <CardHeader>
            <CardTitle className="text-white text-lg">Order Details</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left Column */}
              <div className="space-y-4">
                <div className="flex hover:bg-gray-800/30 p-2 rounded transition-colors duration-200">
                  <span className="text-gray-400 w-32 flex-shrink-0">Number</span>
                  <span className="text-gray-300">:</span>
                  <span className="text-white ml-4 font-mono">{selectedOrder.id}</span>
                </div>
                <div className="flex hover:bg-gray-800/30 p-2 rounded transition-colors duration-200">
                  <span className="text-gray-400 w-32 flex-shrink-0">Customer</span>
                  <span className="text-gray-300">:</span>
                  <span className="text-white ml-4">{selectedOrder.customer}</span>
                </div>
                <div className="flex hover:bg-gray-800/30 p-2 rounded transition-colors duration-200">
                  <span className="text-gray-400 w-32 flex-shrink-0">Phone</span>
                  <span className="text-gray-300">:</span>
                  <span className="text-white ml-4">{selectedOrder.phone}</span>
                </div>
                <div className="flex hover:bg-gray-800/30 p-2 rounded transition-colors duration-200">
                  <span className="text-gray-400 w-32 flex-shrink-0">Address</span>
                  <span className="text-gray-300">:</span>
                  <span className="text-white ml-4">{selectedOrder.address}</span>
                </div>
                <div className="flex hover:bg-gray-800/30 p-2 rounded transition-colors duration-200">
                  <span className="text-gray-400 w-32 flex-shrink-0">Items</span>
                  <span className="text-gray-300">:</span>
                  <span className="text-white ml-4">{selectedOrder.orderItems.length} items</span>
                </div>
              </div>

              {/* Right Column */}
              <div className="space-y-4">
                <div className="flex hover:bg-gray-800/30 p-2 rounded transition-colors duration-200">
                  <span className="text-gray-400 w-32 flex-shrink-0">Status</span>
                  <span className="text-gray-300">:</span>
                  <span className={`ml-4 px-2 py-1 rounded-full text-xs ${getStatusColor(selectedOrder.status)}`}>
                    {selectedOrder.status}
                  </span>
                </div>
                <div className="flex hover:bg-gray-800/30 p-2 rounded transition-colors duration-200">
                  <span className="text-gray-400 w-32 flex-shrink-0">Order Date</span>
                  <span className="text-gray-300">:</span>
                  <span className="text-white ml-4">{selectedOrder.date}</span>
                </div>
                <div className="flex hover:bg-gray-800/30 p-2 rounded transition-colors duration-200">
                  <span className="text-gray-400 w-32 flex-shrink-0">Subtotal</span>
                  <span className="text-gray-300">:</span>
                  <span className="text-white ml-4">${calculateSubtotal(selectedOrder.orderItems)}</span>
                </div>
                <div className="flex hover:bg-gray-800/30 p-2 rounded transition-colors duration-200">
                  <span className="text-gray-400 w-32 flex-shrink-0">Tax (8%)</span>
                  <span className="text-gray-300">:</span>
                  <span className="text-white ml-4">
                    ${calculateTax(calculateSubtotal(selectedOrder.orderItems)).toFixed(2)}
                  </span>
                </div>
                <div className="flex hover:bg-gray-800/30 p-2 rounded transition-colors duration-200">
                  <span className="text-gray-400 w-32 flex-shrink-0">Shipping</span>
                  <span className="text-gray-300">:</span>
                  <span className="text-white ml-4">${calculateShipping()}</span>
                </div>
                <div className="flex border-t border-gray-700 pt-4 hover:bg-gray-800/30 p-2 rounded transition-colors duration-200">
                  <span className="text-gray-400 w-32 flex-shrink-0 font-semibold">Total Cost</span>
                  <span className="text-gray-300">:</span>
                  <span className="text-violet-400 ml-4 font-bold text-lg animate-pulse">
                    ${selectedOrder.netValue}
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Order Items Section */}
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 hover:border-violet-400/50 transition-all duration-300 hover:shadow-lg hover:shadow-violet-500/10 animate-in slide-in-from-right duration-500 delay-200">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-white text-lg">Order Items</CardTitle>
              <p className="text-gray-400 text-sm mt-1">{selectedOrder.customer} order items</p>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="border-gray-600 text-gray-300 hover:text-violet-400 hover:border-violet-400 bg-transparent transition-all duration-200 hover:scale-105"
            >
              <Package className="w-4 h-4 mr-2" />
              Add Item
            </Button>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="text-left py-3 px-2 text-gray-400 font-medium text-sm">ITEM</th>
                    <th className="text-left py-3 px-2 text-gray-400 font-medium text-sm">QUANTITY</th>
                    <th className="text-left py-3 px-2 text-gray-400 font-medium text-sm">UNIT PRICE</th>
                    <th className="text-left py-3 px-2 text-gray-400 font-medium text-sm">TOTAL</th>
                    <th className="text-left py-3 px-2 text-gray-400 font-medium text-sm">STATUS</th>
                    <th className="text-left py-3 px-2 text-gray-400 font-medium text-sm">ACTION</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedOrder.orderItems.map((item, index) => (
                    <tr
                      key={item.id}
                      className="border-b border-gray-800 hover:bg-gray-800/30 transition-all duration-200 hover:scale-[1.01] hover:shadow-sm"
                    >
                      <td className="py-3 px-2">
                        <div>
                          <p className="text-white font-medium">{item.name}</p>
                          <p className="text-gray-400 text-sm">Product ID: {item.id}</p>
                        </div>
                      </td>
                      <td className="py-3 px-2 text-white">{item.quantity} pcs</td>
                      <td className="py-3 px-2 text-white">${item.price}</td>
                      <td className="py-3 px-2 text-violet-400 font-semibold">${item.total}</td>
                      <td className="py-3 px-2">
                        <span className="px-2 py-1 rounded-full text-xs bg-emerald-500/20 text-emerald-400 animate-pulse">
                          Available
                        </span>
                      </td>
                      <td className="py-3 px-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-gray-400 hover:text-white transition-all duration-200 hover:scale-110"
                        >
                          •••
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {/* Order Timeline Section */}
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 hover:border-violet-400/50 transition-all duration-300 hover:shadow-lg hover:shadow-violet-500/10 animate-in slide-in-from-bottom duration-500 delay-300">
          <CardHeader>
            <CardTitle className="text-white text-lg">Order Timeline</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="text-left py-3 px-2 text-gray-400 font-medium text-sm">STAGE</th>
                    <th className="text-left py-3 px-2 text-gray-400 font-medium text-sm">STARTED</th>
                    <th className="text-left py-3 px-2 text-gray-400 font-medium text-sm">COMPLETED</th>
                    <th className="text-left py-3 px-2 text-gray-400 font-medium text-sm">ASSIGNED TO</th>
                    <th className="text-left py-3 px-2 text-gray-400 font-medium text-sm">STATUS</th>
                    <th className="text-left py-3 px-2 text-gray-400 font-medium text-sm">ACTION</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-gray-800 hover:bg-gray-800/30 transition-all duration-200 hover:scale-[1.01]">
                    <td className="py-3 px-2 text-white">Order Processing</td>
                    <td className="py-3 px-2 text-white">{selectedOrder.date}</td>
                    <td className="py-3 px-2 text-white">{selectedOrder.date}</td>
                    <td className="py-3 px-2 text-white">System</td>
                    <td className="py-3 px-2">
                      <span className="px-2 py-1 rounded-full text-xs bg-emerald-500/20 text-emerald-400">
                        Completed
                      </span>
                    </td>
                    <td className="py-3 px-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-gray-400 hover:text-white transition-all duration-200 hover:scale-110"
                      >
                        •••
                      </Button>
                    </td>
                  </tr>
                  <tr className="border-b border-gray-800 hover:bg-gray-800/30 transition-all duration-200 hover:scale-[1.01]">
                    <td className="py-3 px-2 text-white">Payment Verification</td>
                    <td className="py-3 px-2 text-white">{selectedOrder.date}</td>
                    <td className="py-3 px-2 text-white">{selectedOrder.date}</td>
                    <td className="py-3 px-2 text-white">Finance Team</td>
                    <td className="py-3 px-2">
                      <span className="px-2 py-1 rounded-full text-xs bg-emerald-500/20 text-emerald-400">
                        Completed
                      </span>
                    </td>
                    <td className="py-3 px-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-gray-400 hover:text-white transition-all duration-200 hover:scale-110"
                      >
                        •••
                      </Button>
                    </td>
                  </tr>
                  <tr className="border-b border-gray-800 hover:bg-gray-800/30 transition-all duration-200 hover:scale-[1.01]">
                    <td className="py-3 px-2 text-white">Packaging</td>
                    <td className="py-3 px-2 text-white">{selectedOrder.date}</td>
                    <td className="py-3 px-2 text-white">-</td>
                    <td className="py-3 px-2 text-white">Warehouse Team</td>
                    <td className="py-3 px-2">
                      <span
                        className={`px-2 py-1 rounded-full text-xs ${
                          selectedOrder.status === "Processing"
                            ? "bg-blue-500/20 text-blue-400 animate-pulse"
                            : selectedOrder.status === "Completed"
                              ? "bg-emerald-500/20 text-emerald-400"
                              : "bg-yellow-500/20 text-yellow-400 animate-pulse"
                        }`}
                      >
                        {selectedOrder.status === "Completed" ? "Completed" : "In Progress"}
                      </span>
                    </td>
                    <td className="py-3 px-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-gray-400 hover:text-white transition-all duration-200 hover:scale-110"
                      >
                        •••
                      </Button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {selectedOrder.notes && (
          <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 hover:border-violet-400/50 transition-all duration-300 hover:shadow-lg hover:shadow-violet-500/10 animate-in slide-in-from-bottom duration-500 delay-400">
            <CardHeader>
              <CardTitle className="text-white text-lg">Order Notes</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-300 hover:text-white transition-colors duration-200">{selectedOrder.notes}</p>
            </CardContent>
          </Card>
        )}
      </div>
    )
  }

  // Show orders list view
  return (
    <div className="space-y-6 animate-in fade-in-50 duration-500">
      <div className="animate-in slide-in-from-top duration-500">
        <h2 className="text-3xl font-bold text-white mb-2">Orders</h2>
        <p className="text-gray-400">Track and manage customer orders</p>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4 animate-in slide-in-from-top duration-500 delay-100">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4 transition-colors duration-200" />
          <Input
            placeholder="Search orders or customers..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 bg-[#1a1a2e] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300 focus:scale-[1.02]"
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
                  ? "bg-gradient-to-r from-violet-500 to-purple-600 text-white transition-all duration-200 hover:scale-105 shadow-lg hover:shadow-violet-500/25"
                  : "border-gray-600 text-gray-300 hover:text-violet-400 hover:border-violet-400 hover:bg-violet-500/10 transition-all duration-200 hover:scale-105"
              }
            >
              {status}
            </Button>
          ))}
        </div>
      </div>

      {/* Orders Table */}
      <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 hover:border-violet-400/50 transition-all duration-300 hover:shadow-lg hover:shadow-violet-500/10 animate-in slide-in-from-bottom duration-500 delay-200">
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
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Net Value</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Date</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Status</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredOrders.map((order, index) => (
                  <tr
                    key={order.id}
                    onClick={() => handleOrderClick(order)}
                    className="border-b border-gray-800 hover:bg-gray-800/30 transition-all duration-200 cursor-pointer hover:scale-[1.01] hover:shadow-sm animate-in slide-in-from-left duration-300"
                    style={{ animationDelay: `${index * 50}ms` }}
                  >
                    <td className="py-3 px-4 text-emerald-400 font-mono font-semibold hover:text-violet-400 transition-colors duration-200">
                      {order.id}
                    </td>
                    <td className="py-3 px-4 text-white font-medium hover:text-violet-400 transition-colors duration-200">
                      {order.customer}
                    </td>
                    <td className="py-3 px-4 text-violet-400 font-semibold">${order.netValue}</td>
                    <td className="py-3 px-4 text-gray-300">{order.date}</td>
                    <td className="py-3 px-4">
                      <span
                        className={`px-2 py-1 rounded-full text-xs ${getStatusColor(order.status)} transition-all duration-200 hover:scale-110`}
                      >
                        {order.status}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleEditOrder(order)
                        }}
                        className="text-violet-400 hover:text-emerald-400 hover:bg-gray-800/50 transition-all duration-200 hover:scale-110"
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

      {/* Edit Order Modal */}
      {showEditModal && editingOrder && (
        <>
          {/* Modal Overlay */}
          <div
            className="fixed inset-0 bg-black/50 z-50 transition-opacity duration-300 animate-in fade-in"
            onClick={handleCloseEditModal}
          />

          {/* Modal Content */}
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 w-full max-w-4xl max-h-[90vh] overflow-y-auto animate-in zoom-in-95 duration-300">
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-white text-xl">Edit Order {editingOrder.id}</CardTitle>
                  <p className="text-gray-400 text-sm mt-1">Modify order details and items</p>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleCloseEditModal}
                  className="text-gray-400 hover:text-white hover:bg-gray-800/50 transition-all duration-200"
                >
                  <X className="w-5 h-5" />
                </Button>
              </CardHeader>

              <CardContent className="space-y-6">
                {/* Customer Information */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="customer" className="text-violet-400 font-medium">
                      Customer Name
                    </Label>
                    <Input
                      id="customer"
                      value={editingOrder.customer}
                      onChange={(e) => updateEditingOrderField("customer", e.target.value)}
                      className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="phone" className="text-violet-400 font-medium">
                      Phone Number
                    </Label>
                    <Input
                      id="phone"
                      value={editingOrder.phone}
                      onChange={(e) => updateEditingOrderField("phone", e.target.value)}
                      className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20"
                    />
                  </div>
                  <div className="space-y-2 md:col-span-2">
                    <Label htmlFor="address" className="text-violet-400 font-medium">
                      Address
                    </Label>
                    <Input
                      id="address"
                      value={editingOrder.address}
                      onChange={(e) => updateEditingOrderField("address", e.target.value)}
                      className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="status" className="text-violet-400 font-medium">
                      Order Status
                    </Label>
                    <Select
                      value={editingOrder.status}
                      onValueChange={(value) => updateEditingOrderField("status", value)}
                    >
                      <SelectTrigger className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-[#1a1a2e] border-gray-600">
                        <SelectItem value="Pending">Pending</SelectItem>
                        <SelectItem value="Processing">Processing</SelectItem>
                        <SelectItem value="Shipped">Shipped</SelectItem>
                        <SelectItem value="Completed">Completed</SelectItem>
                        <SelectItem value="Cancelled">Cancelled</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="date" className="text-violet-400 font-medium">
                      Order Date
                    </Label>
                    <Input
                      id="date"
                      type="date"
                      value={editingOrder.date}
                      onChange={(e) => updateEditingOrderField("date", e.target.value)}
                      className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20"
                    />
                  </div>
                </div>

                {/* Order Items */}
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-white">Order Items</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-gray-700">
                          <th className="text-left py-2 px-2 text-gray-400 font-medium text-sm">ITEM NAME</th>
                          <th className="text-left py-2 px-2 text-gray-400 font-medium text-sm">QUANTITY</th>
                          <th className="text-left py-2 px-2 text-gray-400 font-medium text-sm">UNIT PRICE</th>
                          <th className="text-left py-2 px-2 text-gray-400 font-medium text-sm">TOTAL</th>
                        </tr>
                      </thead>
                      <tbody>
                        {editingOrder.orderItems.map((item) => (
                          <tr key={item.id} className="border-b border-gray-800">
                            <td className="py-2 px-2">
                              <Input
                                value={item.name}
                                onChange={(e) => updateOrderItem(item.id, "name", e.target.value)}
                                className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 text-sm"
                              />
                            </td>
                            <td className="py-2 px-2">
                              <Input
                                type="number"
                                value={item.quantity}
                                onChange={(e) =>
                                  updateOrderItem(item.id, "quantity", Number.parseInt(e.target.value) || 0)
                                }
                                className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 text-sm w-20"
                              />
                            </td>
                            <td className="py-2 px-2">
                              <Input
                                type="number"
                                value={item.price}
                                onChange={(e) =>
                                  updateOrderItem(item.id, "price", Number.parseFloat(e.target.value) || 0)
                                }
                                className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 text-sm w-24"
                              />
                            </td>
                            <td className="py-2 px-2">
                              <span className="text-violet-400 font-semibold">${item.total}</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Order Summary */}
                <div className="bg-[#0f0f23] p-4 rounded-lg border border-gray-700">
                  <h3 className="text-lg font-semibold text-white mb-3">Order Summary</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Subtotal:</span>
                      <span className="text-white">${calculateSubtotal(editingOrder.orderItems)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Tax (8%):</span>
                      <span className="text-white">
                        ${calculateTax(calculateSubtotal(editingOrder.orderItems)).toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Shipping:</span>
                      <span className="text-white">${calculateShipping()}</span>
                    </div>
                    <div className="flex justify-between border-t border-gray-700 pt-2">
                      <span className="text-white font-semibold">Total:</span>
                      <span className="text-violet-400 font-bold text-lg">
                        $
                        {Math.round(
                          calculateSubtotal(editingOrder.orderItems) +
                            calculateTax(calculateSubtotal(editingOrder.orderItems)) +
                            calculateShipping(),
                        )}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Notes */}
                <div className="space-y-2">
                  <Label htmlFor="notes" className="text-violet-400 font-medium">
                    Order Notes
                  </Label>
                  <Textarea
                    id="notes"
                    value={editingOrder.notes || ""}
                    onChange={(e) => updateEditingOrderField("notes", e.target.value)}
                    className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20"
                    rows={3}
                    placeholder="Add any notes about this order..."
                  />
                </div>

                {/* Action Buttons */}
                <div className="flex justify-end space-x-3 pt-4 border-t border-gray-700">
                  <Button
                    variant="outline"
                    onClick={handleCloseEditModal}
                    className="border-gray-600 text-gray-300 hover:text-white hover:border-gray-500 bg-transparent"
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={handleSaveOrder}
                    className="bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white transition-all duration-200 hover:scale-105 shadow-lg hover:shadow-violet-500/25"
                  >
                    <Save className="w-4 h-4 mr-2" />
                    Save Changes
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </div>
  )
}
