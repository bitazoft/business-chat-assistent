"use client"

import { useState, useEffect } from "react"
import { Search, Eye, ArrowLeft, Package, X, Save, Edit, ChevronDown, Trash2, Plus } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { getCurrentUser } from "@/lib/auth"
import { getAuth } from "@/lib/authUtils"
import { toast } from "sonner"
import { 
  fetchOrders, 
  fetchProducts, 
  updateOrderStatus, 
  updatePaymentStatus, 
  updateOrder,
  addProductToOrder as addProductToOrderService,
  removeOrderItem as removeOrderItemService,
  copyToClipboard,
  type Order as ServiceOrder,
  type OrderItem as ServiceOrderItem,
  type Product as ServiceProduct
} from "@/services/orderService"

// Use types from the service
type OrderItem = ServiceOrderItem
type Product = ServiceProduct  
type Order = ServiceOrder

export function OrdersPage() {
  const [orders, setOrders] = useState<Order[]>([])
  const [products, setProducts] = useState<Product[]>([])
  const [searchTerm, setSearchTerm] = useState("")
  const [statusFilter, setStatusFilter] = useState("All")
  const [selectedOrder, setSelectedOrder] = useState<Order | null>(null)
  const [showOrderDetails, setShowOrderDetails] = useState(false)
  const [showEditModal, setShowEditModal] = useState(false)
  const [showProductModal, setShowProductModal] = useState(false)
  const [editingOrder, setEditingOrder] = useState<Order | null>(null)
  const [loading, setLoading] = useState(true)
  const user = getCurrentUser()

  useEffect(() => {
    loadOrders()
    loadProducts()
  }, [])

  const loadOrders = async () => {
    if (!user?.sellerId) {
      toast.error("User not authenticated")
      setLoading(false)
      return
    }

    try {
      const ordersData = await fetchOrders(user.sellerId.toString())
      setOrders(ordersData)
    } catch (error) {
      // Error handling is already done in the service
    } finally {
      setLoading(false)
    }
  }

  const loadProducts = async () => {
    if (!user?.sellerId) {
      console.error("No user or sellerId found")
      return
    }

    try {
      const productsData = await fetchProducts(user.sellerId.toString())
      setProducts(productsData)
    } catch (error) {
      // Error handling is already done in the service
    }
  }

  const refreshOrders = () => {
    setLoading(true)
    loadOrders()
  }

  const filteredOrders = orders.filter((order) => {
    const customer = order.customer || ''
    const orderId = order.id || ''
    const status = order.status || ''
    
    const matchesSearch =
      customer.toLowerCase().includes(searchTerm.toLowerCase()) ||
      orderId.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesStatus = statusFilter === "All" || status === statusFilter
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

  const getPaymentStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case "paid":
      case "completed":
        return "bg-emerald-500/20 text-emerald-400"
      case "pending":
        return "bg-yellow-500/20 text-yellow-400"
      case "failed":
      case "rejected":
        return "bg-red-500/20 text-red-400"
      case "processing":
        return "bg-blue-500/20 text-blue-400"
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

  const handleEditOrder = (order: Order | null) => {
    if (!order) {
      toast.error("No order selected")
      return
    }
    handleBackToOrders()
    console.log("Opening edit modal for order:", order)
    setEditingOrder({ ...order })
    setShowEditModal(true)
  }

  const handleUpdateOrderStatus = async (orderId: string, newStatus: string) => {
    try {
      await updateOrderStatus(orderId, newStatus)
      
      // Refresh orders to get the updated data
      await loadOrders()

      // Update selected order if it's the same one being updated
      if (selectedOrder && selectedOrder.id === orderId) {
        setSelectedOrder({...selectedOrder, status: newStatus})
      }
    } catch (error) {
      // Error handling is already done in the service
    }
  }

  const handleUpdatePaymentStatus = async (orderId: string, newPaymentStatus: string) => {
    try {
      await updatePaymentStatus(orderId, newPaymentStatus)
      
      // Refresh orders to get the updated data
      await loadOrders()

      // Update selected order if it's the same one being updated
      if (selectedOrder && selectedOrder.id === orderId) {
        setSelectedOrder({...selectedOrder, paymentStatus: newPaymentStatus})
      }
    } catch (error) {
      // Error handling is already done in the service
    }
  }

  const handleSaveOrder = async () => {
    if (!editingOrder) return

    try {
      await updateOrder(editingOrder)
      
      // Refresh orders to get the updated data
      await loadOrders()

      // Update selected order if it's the same one being edited
      if (selectedOrder && selectedOrder.id === editingOrder.id) {
        setSelectedOrder({...editingOrder})
      }

      setShowEditModal(false)
      setEditingOrder(null)
    } catch (error) {
      // Error handling is already done in the service
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

  const addOrderItem = () => {
    setShowProductModal(true)
  }

  const addProductToOrder = async (product: Product, quantity: number = 1) => {
    if (!editingOrder) return

    try {
      const data = await addProductToOrderService(
        editingOrder.id, 
        product.id, 
        quantity, 
        product.price
      )
      
      // Transform the updated order data
      const transformedOrder = {
        ...editingOrder,
        orderItems: (data.order.order_items || []).map((item: any) => ({
          id: item.id?.toString() || '',
          name: item.products?.name || item.product_name || item.name || 'Unknown Product',
          productId: item.product_id?.toString() || '',
          quantity: item.quantity || 0,
          price: item.price || 0,
          total: (item.quantity * item.price) || 0
        }))
      }

      setEditingOrder(transformedOrder)
      
      // Update the selected order if it's the same one
      if (selectedOrder && selectedOrder.id === editingOrder.id) {
        setSelectedOrder(transformedOrder)
      }

      setShowProductModal(false)
      
      // Refresh orders to get updated data
      await loadOrders()
    } catch (error) {
      // Error handling is already done in the service
    }
  }

  const removeOrderItem = async (itemId: string) => {
    if (!editingOrder) return

    try {
      // Don't try to remove items that haven't been saved yet (temporary items)
      if (itemId.startsWith('temp_') || itemId === '-1') {
        // Just remove from local state for temporary items
        const updatedItems = editingOrder.orderItems.filter((item: OrderItem) => item.id !== itemId)
        
        setEditingOrder({
          ...editingOrder,
          orderItems: updatedItems
        })
        
        toast.success("Item removed from order")
        return
      }

      const data = await removeOrderItemService(editingOrder.id, itemId)
      
      // Transform the updated order data
      const transformedOrder = {
        ...editingOrder,
        orderItems: (data.order.order_items || []).map((item: any) => ({
          id: item.id?.toString() || '',
          name: item.products?.name || item.product_name || item.name || 'Unknown Product',
          productId: item.product_id?.toString() || '',
          quantity: item.quantity || 0,
          price: item.price || 0,
          total: (item.quantity * item.price) || 0
        }))
      }

      setEditingOrder(transformedOrder)
      
      // Update the selected order if it's the same one
      if (selectedOrder && selectedOrder.id === editingOrder.id) {
        setSelectedOrder(transformedOrder)
      }
      
      // Refresh orders to get updated data
      await loadOrders()
    } catch (error) {
      // Error handling is already done in the service
    }
  }

  const calculateSubtotal = (items: OrderItem[]) => {
    return items.reduce((sum, item) => sum + item.price, 0)
  }

  const calculateTax = (subtotal: number) => {
    return subtotal * 0.08 // 8% tax
  }

  const calculateShipping = () => {
    return selectedOrder?.shippingCost || 0 // Use order's shipping cost
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
              {/* <Button
                variant="outline"
                size="sm"
                className="border-gray-600 text-gray-300 hover:text-violet-400 hover:border-violet-400 bg-transparent transition-all duration-200 hover:scale-105"
              >
                Export
              </Button> */}
              
              {/* Payment Status Dropdown */}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    size="sm"
                    className="bg-blue-500 hover:bg-blue-600 text-white transition-all duration-200 hover:scale-105 hover:shadow-lg hover:shadow-blue-500/25"
                  >
                    Payment Status
                    <ChevronDown className="w-4 h-4 ml-2" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="bg-[#1a1a2e] border-gray-600 text-white">
                  <DropdownMenuItem 
                    onClick={() => handleUpdatePaymentStatus(selectedOrder.id, "Paid")}
                    className="hover:bg-gray-800 cursor-pointer"
                  >
                    Mark as Paid
                  </DropdownMenuItem>
                  <DropdownMenuItem 
                    onClick={() => handleUpdatePaymentStatus(selectedOrder.id, "Pending")}
                    className="hover:bg-gray-800 cursor-pointer"
                  >
                    Mark as Pending
                  </DropdownMenuItem>
                  <DropdownMenuItem 
                    onClick={() => handleUpdatePaymentStatus(selectedOrder.id, "Processing")}
                    className="hover:bg-gray-800 cursor-pointer"
                  >
                    Mark as Processing
                  </DropdownMenuItem>
                  <DropdownMenuItem 
                    onClick={() => handleUpdatePaymentStatus(selectedOrder.id, "Failed")}
                    className="hover:bg-gray-800 cursor-pointer"
                  >
                    Mark as Failed
                  </DropdownMenuItem>
                  <DropdownMenuItem 
                    onClick={() => handleUpdatePaymentStatus(selectedOrder.id, "Rejected")}
                    className="hover:bg-gray-800 cursor-pointer"
                  >
                    Mark as Rejected
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>

              {/* Order Status Dropdown */}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    size="sm"
                    className="bg-emerald-500 hover:bg-emerald-600 text-white transition-all duration-200 hover:scale-105 hover:shadow-lg hover:shadow-emerald-500/25"
                  >
                    Order Status
                    <ChevronDown className="w-4 h-4 ml-2" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="bg-[#1a1a2e] border-gray-600 text-white">
                  <DropdownMenuItem 
                    onClick={() => handleUpdateOrderStatus(selectedOrder.id, "Pending")}
                    className="hover:bg-gray-800 cursor-pointer"
                  >
                    Mark as Pending
                  </DropdownMenuItem>
                  <DropdownMenuItem 
                    onClick={() => handleUpdateOrderStatus(selectedOrder.id, "Processing")}
                    className="hover:bg-gray-800 cursor-pointer"
                  >
                    Mark as Processing
                  </DropdownMenuItem>
                  <DropdownMenuItem 
                    onClick={() => handleUpdateOrderStatus(selectedOrder.id, "Shipped")}
                    className="hover:bg-gray-800 cursor-pointer"
                  >
                    Mark as Shipped
                  </DropdownMenuItem>
                  <DropdownMenuItem 
                    onClick={() => handleUpdateOrderStatus(selectedOrder.id, "Completed")}
                    className="hover:bg-gray-800 cursor-pointer"
                  >
                    Mark as Completed
                  </DropdownMenuItem>
                  <DropdownMenuItem 
                    onClick={() => handleUpdateOrderStatus(selectedOrder.id, "Cancelled")}
                    className="hover:bg-gray-800 cursor-pointer"
                  >
                    Mark as Cancelled
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
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
                  <div className="ml-4 flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded-full text-xs ${getStatusColor(selectedOrder.status)}`}>
                      {selectedOrder.status}
                    </span>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          size="sm"
                          variant="outline"
                          className="border-gray-600 text-gray-400 hover:text-white hover:border-violet-400 bg-transparent transition-all duration-200 text-xs px-2 py-1 h-6"
                        >
                          Change
                          <ChevronDown className="w-3 h-3 ml-1" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent className="bg-[#1a1a2e] border-gray-600 text-white">
                        <DropdownMenuItem 
                          onClick={() => handleUpdateOrderStatus(selectedOrder.id, "Pending")}
                          className="hover:bg-gray-800 cursor-pointer text-xs"
                        >
                          Pending
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          onClick={() => handleUpdateOrderStatus(selectedOrder.id, "Processing")}
                          className="hover:bg-gray-800 cursor-pointer text-xs"
                        >
                          Processing
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          onClick={() => handleUpdateOrderStatus(selectedOrder.id, "Shipped")}
                          className="hover:bg-gray-800 cursor-pointer text-xs"
                        >
                          Shipped
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          onClick={() => handleUpdateOrderStatus(selectedOrder.id, "Completed")}
                          className="hover:bg-gray-800 cursor-pointer text-xs"
                        >
                          Completed
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          onClick={() => handleUpdateOrderStatus(selectedOrder.id, "Cancelled")}
                          className="hover:bg-gray-800 cursor-pointer text-xs"
                        >
                          Cancelled
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </div>
                <div className="flex hover:bg-gray-800/30 p-2 rounded transition-colors duration-200">
                  <span className="text-gray-400 w-32 flex-shrink-0">Order Date</span>
                  <span className="text-gray-300">:</span>
                  <span className="text-white ml-4">{selectedOrder.date}</span>
                </div>
                <div className="flex hover:bg-gray-800/30 p-2 rounded transition-colors duration-200">
                  <span className="text-gray-400 w-32 flex-shrink-0">Subtotal</span>
                  <span className="text-gray-300">:</span>
                  <span className="text-white ml-4">Rs.{selectedOrder.netValue.toString()}</span>
                </div>
                {/* <div className="flex hover:bg-gray-800/30 p-2 rounded transition-colors duration-200">
                  <span className="text-gray-400 w-32 flex-shrink-0">Tax (8%)</span>
                  <span className="text-gray-300">:</span>
                  <span className="text-white ml-4">
                    ${calculateTax(calculateSubtotal(selectedOrder.orderItems)).toFixed(2)}
                  </span>
                </div> */}
                <div className="flex hover:bg-gray-800/30 p-2 rounded transition-colors duration-200">
                  <span className="text-gray-400 w-32 flex-shrink-0">Shipping</span>
                  <span className="text-gray-300">:</span>
                  <span className="text-white ml-4">Rs.{calculateShipping()}</span>
                </div>
                <div className="flex border-t border-gray-700 pt-4 hover:bg-gray-800/30 p-2 rounded transition-colors duration-200">
                  <span className="text-gray-400 w-32 flex-shrink-0 font-semibold">Total Cost</span>
                  <span className="text-gray-300">:</span>
                  <span className="text-violet-400 ml-4 font-bold text-lg animate-pulse">
                    {(parseFloat(selectedOrder.netValue.toString()) + parseFloat(calculateShipping().toString())).toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Payment Information Section */}
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 hover:border-violet-400/50 transition-all duration-300 hover:shadow-lg hover:shadow-violet-500/10 animate-in slide-in-from-left duration-500 delay-150">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-white text-lg">Payment Information</CardTitle>
            <div className="flex items-center space-x-2">
              <Button
                size="sm"
                variant="outline"
                onClick={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  console.log("Edit Payment button clicked", selectedOrder)
                  if (selectedOrder) {
                    console.log("Calling handleEditOrder with:", selectedOrder)
                    handleEditOrder(selectedOrder)
                  } else {
                    console.error("selectedOrder is null")
                    toast.error("No order selected")
                  }
                }}
                className="border-violet-400 text-violet-400 hover:bg-violet-400 hover:text-white transition-all duration-200 text-xs px-3 py-1 h-7"
              >
                Edit Payment
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="flex hover:bg-gray-800/30 p-2 rounded transition-colors duration-200">
                  <span className="text-gray-400 w-32 flex-shrink-0">Payment Status</span>
                  <span className="text-gray-300">:</span>
                  <div className="ml-4 flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded-full text-xs ${getPaymentStatusColor(selectedOrder.paymentStatus || 'Pending')}`}>
                      {selectedOrder.paymentStatus || 'Pending'}
                    </span>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          size="sm"
                          variant="outline"
                          className="border-gray-600 text-gray-400 hover:text-white hover:border-violet-400 bg-transparent transition-all duration-200 text-xs px-2 py-1 h-6"
                        >
                          Change
                          <ChevronDown className="w-3 h-3 ml-1" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent className="bg-[#1a1a2e] border-gray-600 text-white">
                        <DropdownMenuItem 
                          onClick={() => handleUpdatePaymentStatus(selectedOrder.id, "Paid")}
                          className="hover:bg-gray-800 cursor-pointer text-xs"
                        >
                          Paid
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          onClick={() => handleUpdatePaymentStatus(selectedOrder.id, "Pending")}
                          className="hover:bg-gray-800 cursor-pointer text-xs"
                        >
                          Pending
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          onClick={() => handleUpdatePaymentStatus(selectedOrder.id, "Processing")}
                          className="hover:bg-gray-800 cursor-pointer text-xs"
                        >
                          Processing
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          onClick={() => handleUpdatePaymentStatus(selectedOrder.id, "Failed")}
                          className="hover:bg-gray-800 cursor-pointer text-xs"
                        >
                          Failed
                        </DropdownMenuItem>
                        <DropdownMenuItem 
                          onClick={() => handleUpdatePaymentStatus(selectedOrder.id, "Rejected")}
                          className="hover:bg-gray-800 cursor-pointer text-xs"
                        >
                          Rejected
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </div>
                <div className="flex hover:bg-gray-800/30 p-2 rounded transition-colors duration-200">
                  <span className="text-gray-400 w-32 flex-shrink-0">Payment Method</span>
                  <span className="text-gray-300">:</span>
                  <span className="text-white ml-4">{selectedOrder.paymentMethod || 'Unknown'}</span>
                </div>
                <div className="flex hover:bg-gray-800/30 p-2 rounded transition-colors duration-200">
                  <span className="text-gray-400 w-32 flex-shrink-0">Amount Paid</span>
                  <span className="text-gray-300">:</span>
                  <span className="text-emerald-400 ml-4 font-semibold">
                  Rs.{(parseFloat(selectedOrder.netValue.toString()) + parseFloat(calculateShipping().toString())).toFixed(2)}
                  </span>
                </div>
              </div>
              <div className="space-y-4">
                <div className="hover:bg-gray-800/30 p-2 rounded transition-colors duration-200">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-gray-400 font-medium">Payment Proof</span>
                    {selectedOrder.paymentProofUrl && (
                      <div className="flex items-center space-x-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => window.open(selectedOrder.paymentProofUrl, '_blank')}
                          className="border-violet-400 text-violet-400 hover:bg-violet-400 hover:text-white transition-all duration-200 text-xs px-2 py-1 h-6"
                        >
                          View Full Size
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => copyToClipboard(selectedOrder.paymentProofUrl || '', 'Payment proof URL')}
                          className="border-blue-400 text-blue-400 hover:bg-blue-400 hover:text-white transition-all duration-200 text-xs px-2 py-1 h-6"
                        >
                          Copy URL
                        </Button>
                      </div>
                    )}
                  </div>
                  {selectedOrder.paymentProofUrl ? (
                    <div className="space-y-2">
                      <a 
                        href={selectedOrder.paymentProofUrl} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="text-violet-400 hover:text-violet-300 underline transition-colors duration-200 font-medium text-sm block"
                      >
                        {selectedOrder.paymentProofUrl.length > 50 
                          ? `${selectedOrder.paymentProofUrl.substring(0, 50)}...` 
                          : selectedOrder.paymentProofUrl}
                      </a>
                      <div className="relative">
                        <img 
                          src={selectedOrder.paymentProofUrl} 
                          alt="Payment Proof" 
                          className="w-full max-w-sm h-40 object-cover rounded-lg border border-gray-600 hover:border-violet-400 transition-all duration-200 cursor-pointer hover:scale-105"
                          onClick={() => window.open(selectedOrder.paymentProofUrl, '_blank')}
                          onError={(e) => {
                            e.currentTarget.style.display = 'none';
                            const errorDiv = document.createElement('div');
                            errorDiv.className = 'w-full max-w-sm h-40 bg-gray-800 border border-gray-600 rounded-lg flex items-center justify-center text-gray-400 text-sm';
                            errorDiv.textContent = 'Unable to load image preview';
                            e.currentTarget.parentNode?.appendChild(errorDiv);
                          }}
                        />
                      </div>
                    </div>
                  ) : (
                    <div className="w-full max-w-sm h-40 bg-gray-800 border-2 border-dashed border-gray-600 rounded-lg flex flex-col items-center justify-center text-gray-400 text-sm">
                      <Package className="w-8 h-8 mb-2 opacity-50" />
                      <span>No payment proof uploaded</span>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={(e) => {
                          e.preventDefault()
                          e.stopPropagation()
                          console.log("Add Proof URL button clicked", selectedOrder)
                          if (selectedOrder) {
                            console.log("Calling handleEditOrder with:", selectedOrder)
                            handleEditOrder(selectedOrder)
                          } else {
                            console.error("selectedOrder is null")
                            toast.error("No order selected")
                          }
                        }}
                        className="border-violet-400 text-violet-400 hover:bg-violet-400 hover:text-white transition-all duration-200 text-xs px-3 py-1 h-6 mt-2"
                      >
                        Add Proof URL
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Order Items Section */}
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 hover:border-violet-400/50 transition-all duration-300 hover:shadow-lg hover:shadow-violet-500/10 animate-in slide-in-from-right duration-500 delay-300">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-white text-lg">Order Items</CardTitle>
              <p className="text-gray-400 text-sm mt-1">{selectedOrder.customer}'s order items</p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                console.log("Add Item button clicked")
                if (selectedOrder) {
                  // Open edit modal and add a new item
                  handleBackToOrders()
                  const tempOrder = { ...selectedOrder }
                  setEditingOrder(tempOrder)
                  setShowEditModal(true)
                  // Add new item after modal opens
                  setTimeout(() => {
                    addOrderItem()
                  }, 100)
                } else {
                  toast.error("No order selected")
                }
              }}
              className="border-gray-600 text-gray-300 hover:text-violet-400 hover:border-violet-400 bg-transparent transition-all duration-200 hover:scale-105"
            >
              <Plus className="w-4 h-4 mr-2" />
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
                      <td className="py-3 px-2 text-white">Rs.{item.price}</td>
                      <td className="py-3 px-2 text-violet-400 font-semibold">Rs.{item.price * item.quantity}</td>
                      <td className="py-3 px-2">
                        <span className="px-2 py-1 rounded-full text-xs bg-emerald-500/20 text-emerald-400 animate-pulse">
                          Available
                        </span>
                      </td>
                      <td className="py-3 px-2">
                        <div className="flex items-center space-x-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => {
                              if (selectedOrder) {
                                handleEditOrder(selectedOrder)
                              }
                            }}
                            className="text-gray-400 hover:text-violet-400 transition-all duration-200 hover:scale-110"
                            title="Edit Item"
                          >
                            <Edit className="w-3 h-3" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => {
                              // This will be handled in the edit modal
                              if (selectedOrder) {
                                handleEditOrder(selectedOrder)
                              }
                            }}
                            className="text-gray-400 hover:text-red-400 transition-all duration-200 hover:scale-110"
                            title="Remove Item (Edit in modal)"
                          >
                            •••
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
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

  // Product Selection Modal Component
  const ProductSelectionModal = () => {
    const [selectedProduct, setSelectedProduct] = useState<Product | null>(null)
    const [quantity, setQuantity] = useState(1)

    const handleAddProduct = () => {
      if (selectedProduct && quantity > 0) {
        addProductToOrder(selectedProduct, quantity)
        setSelectedProduct(null)
        setQuantity(1)
      }
    }

    return showProductModal ? (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 w-full max-w-2xl mx-4 max-h-[80vh] overflow-y-auto">
          <CardHeader>
            <div className="flex justify-between items-center">
              <CardTitle className="text-white">Select Product to Add</CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowProductModal(false)}
                className="text-gray-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {products.length === 0 ? (
              <p className="text-gray-400 text-center py-8">No products available</p>
            ) : (
              <>
                <div className="space-y-2">
                  <Label className="text-violet-400">Select Product</Label>
                  <Select
                    value={selectedProduct?.id || ""}
                    onValueChange={(value) => {
                      const product = products.find(p => p.id === value)
                      setSelectedProduct(product || null)
                    }}
                  >
                    <SelectTrigger className="bg-gray-800 border-gray-600 text-white">
                      <SelectValue placeholder="Choose a product..." />
                    </SelectTrigger>
                    <SelectContent className="bg-gray-800 border-gray-600">
                      {products.map((product) => (
                        <SelectItem key={product.id} value={product.id} className="text-white hover:bg-gray-700">
                          <div className="flex justify-between items-center w-full">
                            <span>{product.name}</span>
                            <span className="text-emerald-400 ml-2">Rs.{product.price}</span>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {selectedProduct && (
                  <div className="bg-gray-800/50 p-4 rounded-lg space-y-3">
                    <h4 className="text-white font-semibold">{selectedProduct.name}</h4>
                    <p className="text-gray-300">Price: <span className="text-emerald-400">Rs.{selectedProduct.price}</span></p>
                    <p className="text-gray-300">Stock: <span className="text-blue-400">{selectedProduct.stock}</span></p>
                    {selectedProduct.description && (
                      <p className="text-gray-400 text-sm">{selectedProduct.description}</p>
                    )}
                  </div>
                )}

                <div className="space-y-2">
                  <Label className="text-violet-400">Quantity</Label>
                  <Input
                    type="number"
                    min="1"
                    max={selectedProduct?.stock || 1}
                    value={quantity}
                    onChange={(e) => setQuantity(parseInt(e.target.value) || 1)}
                    className="bg-gray-800 border-gray-600 text-white"
                  />
                </div>

                <div className="flex justify-end space-x-2 pt-4">
                  <Button
                    variant="outline"
                    onClick={() => setShowProductModal(false)}
                    className="border-gray-600 text-gray-300 hover:text-white hover:border-gray-400"
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={handleAddProduct}
                    disabled={!selectedProduct || quantity <= 0}
                    className="bg-gradient-to-r from-violet-500 to-purple-600 text-white hover:scale-105"
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    Add to Order
                  </Button>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    ) : null
  }

  // Show orders list view
  return (
    <div className="space-y-6 animate-in fade-in-50 duration-500">
      <div className="flex justify-between items-center animate-in slide-in-from-top duration-500">
        <div>
          <h2 className="text-3xl font-bold text-white mb-2">Orders</h2>
          <p className="text-gray-400">Track and manage customer orders</p>
        </div>
        <Button
          onClick={refreshOrders}
          disabled={loading}
          className="bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white shadow-lg hover:shadow-violet-500/25 transition-all duration-300"
        >
          {loading ? (
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
          ) : (
            <Package className="w-4 h-4 mr-2" />
          )}
          Refresh Orders
        </Button>
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
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Shipping</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Payment Status</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Payment Method</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Date</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Status</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={9} className="py-12 text-center">
                      <div className="flex flex-col items-center justify-center">
                        <div className="w-8 h-8 border-2 border-violet-400 border-t-transparent rounded-full animate-spin mb-4"></div>
                        <p className="text-gray-400">Loading orders...</p>
                      </div>
                    </td>
                  </tr>
                ) : filteredOrders.length === 0 ? (
                  <tr>
                    <td colSpan={9} className="py-12 text-center">
                      <div className="flex flex-col items-center justify-center">
                        <Package className="w-16 h-16 text-gray-600 mb-4" />
                        <h3 className="text-xl font-semibold text-gray-400 mb-2">No orders found</h3>
                        <p className="text-gray-500">
                          {searchTerm || statusFilter !== "All" 
                            ? "No orders match your search criteria." 
                            : "You don't have any orders yet."}
                        </p>
                      </div>
                    </td>
                  </tr>
                ) : (
                  filteredOrders.map((order: Order, index: number) => (
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
                      <td className="py-3 px-4 text-violet-400 font-semibold">Rs.{order.netValue}</td>
                      <td className="py-3 px-4 text-emerald-400 font-semibold">Rs.{order.shippingCost}</td>
                      <td className="py-3 px-4">
                        <span
                          className={`px-2 py-1 rounded-full text-xs ${getPaymentStatusColor(order.paymentStatus || 'Pending')} transition-all duration-200 hover:scale-110`}
                        >
                          {order.paymentStatus || 'Pending'}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-gray-300">{order.paymentMethod || 'Unknown'}</td>
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
                  ))
                )}
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
                  <div className="space-y-2">
                    <Label htmlFor="email" className="text-violet-400 font-medium">
                      Email Address
                    </Label>
                    <Input
                      id="email"
                      type="email"
                      value={editingOrder.email || ''}
                      onChange={(e) => updateEditingOrderField("email", e.target.value)}
                      className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20"
                      placeholder="customer@example.com"
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
                    <Label htmlFor="paymentStatus" className="text-violet-400 font-medium">
                      Payment Status
                    </Label>
                    <Select
                      value={editingOrder.paymentStatus || 'Pending'}
                      onValueChange={(value) => updateEditingOrderField("paymentStatus", value)}
                    >
                      <SelectTrigger className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-[#1a1a2e] border-gray-600">
                        <SelectItem value="Pending">Pending</SelectItem>
                        <SelectItem value="Paid">Paid</SelectItem>
                        <SelectItem value="Failed">Failed</SelectItem>
                        <SelectItem value="Processing">Processing</SelectItem>
                        <SelectItem value="Rejected">Rejected</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="paymentMethod" className="text-violet-400 font-medium">
                      Payment Method
                    </Label>
                    <Select
                      value={editingOrder.paymentMethod || 'Unknown'}
                      onValueChange={(value) => updateEditingOrderField("paymentMethod", value)}
                    >
                      <SelectTrigger className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-[#1a1a2e] border-gray-600">
                        <SelectItem value="Cash">Cash</SelectItem>
                        <SelectItem value="Credit Card">Credit Card</SelectItem>
                        <SelectItem value="Debit Card">Debit Card</SelectItem>
                        <SelectItem value="Bank Transfer">Bank Transfer</SelectItem>
                        <SelectItem value="Digital Wallet">Digital Wallet</SelectItem>
                        <SelectItem value="PayPal">PayPal</SelectItem>
                        <SelectItem value="Cryptocurrency">COD</SelectItem>
                        <SelectItem value="Unknown">Unknown</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="paymentProofUrl" className="text-violet-400 font-medium">
                      Payment Proof URL
                    </Label>
                    <Input
                      id="paymentProofUrl"
                      type="url"
                      value={editingOrder.paymentProofUrl || ''}
                      onChange={(e) => updateEditingOrderField("paymentProofUrl", e.target.value)}
                      className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20"
                      placeholder="https://example.com/payment-proof.jpg"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="shippingCost" className="text-violet-400 font-medium">
                      Shipping Cost (Rs.)
                    </Label>
                    <Input
                      id="shippingCost"
                      type="number"
                      step="0.01"
                      value={editingOrder.shippingCost}
                      onChange={(e) => updateEditingOrderField("shippingCost", Number.parseFloat(e.target.value) || 0)}
                      className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20"
                      placeholder="0.00"
                    />
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
                </div>                {/* Order Items */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-white">Order Items</h3>
                    <Button
                      size="sm"
                      onClick={addOrderItem}
                      className="bg-violet-500 hover:bg-violet-600 text-white transition-all duration-200 hover:scale-105"
                    >
                      <Plus className="w-4 h-4 mr-2" />
                      Add Item
                    </Button>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-gray-700">
                          <th className="text-left py-2 px-2 text-gray-400 font-medium text-sm">ITEM NAME</th>
                          <th className="text-left py-2 px-2 text-gray-400 font-medium text-sm">QUANTITY</th>
                          <th className="text-left py-2 px-2 text-gray-400 font-medium text-sm">UNIT PRICE</th>
                          <th className="text-left py-2 px-2 text-gray-400 font-medium text-sm">TOTAL</th>
                          <th className="text-left py-2 px-2 text-gray-400 font-medium text-sm">ACTION</th>
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
                                placeholder="Enter item name"
                              />
                            </td>
                            <td className="py-2 px-2">
                              <Input
                                type="number"
                                min="1"
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
                                min="0"
                                step="0.01"
                                value={item.price}
                                onChange={(e) =>
                                  updateOrderItem(item.id, "price", Number.parseFloat(e.target.value) || 0)
                                }
                                className="bg-[#0f0f23] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 text-sm w-24"
                              />
                            </td>
                            <td className="py-2 px-2">
                              <span className="text-violet-400 font-semibold">Rs.{item.total.toFixed(2)}</span>
                            </td>
                            <td className="py-2 px-2">
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => removeOrderItem(item.id)}
                                disabled={editingOrder.orderItems.length <= 1}
                                className="text-red-400 hover:text-red-300 hover:bg-red-400/10 transition-all duration-200 hover:scale-110 disabled:opacity-50 disabled:cursor-not-allowed"
                                title={editingOrder.orderItems.length <= 1 ? "Cannot remove the last item" : "Remove item"}
                              >
                                <Trash2 className="w-4 h-4" />
                              </Button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {editingOrder.orderItems.length === 0 && (
                    <div className="text-center py-8 text-gray-400">
                      <Package className="w-12 h-12 mx-auto mb-4 opacity-50" />
                      <p>No items in this order</p>
                      <Button
                        size="sm"
                        onClick={addOrderItem}
                        className="bg-violet-500 hover:bg-violet-600 text-white transition-all duration-200 hover:scale-105 mt-2"
                      >
                        <Plus className="w-4 h-4 mr-2" />
                        Add First Item
                      </Button>
                    </div>
                  )}
                </div>

                {/* Order Summary */}
                <div className="bg-[#0f0f23] p-4 rounded-lg border border-gray-700">
                  <h3 className="text-lg font-semibold text-white mb-3">Order Summary</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Subtotal:</span>
                      <span className="text-white">Rs.{calculateSubtotal(editingOrder.orderItems)}</span>
                    </div>
                    {/* <div className="flex justify-between">
                      <span className="text-gray-400">Tax (8%):</span>
                      <span className="text-white">
                        ${calculateTax(calculateSubtotal(editingOrder.orderItems)).toFixed(2)}
                      </span>
                    </div> */}
                    <div className="flex justify-between">
                      <span className="text-gray-400">Shipping:</span>
                      <span className="text-white">Rs.{calculateShipping()}</span>
                    </div>
                    <div className="flex justify-between border-t border-gray-700 pt-2">
                      <span className="text-white font-semibold">Total:</span>
                      <span className="text-violet-400 font-bold text-lg">
                        Rs.
                        {Math.round(
                          calculateSubtotal(editingOrder.orderItems) + calculateShipping(),
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

      {/* Product Selection Modal */}
      <ProductSelectionModal />
    </div>
  )
}
