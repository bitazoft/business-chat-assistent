"use client"

import { useState, useEffect } from "react"
import { toast } from "sonner"
import { Edit, Trash2, Search, Users, Eye, Ban, CheckCircle, UserPlus, Filter, Download } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { AddUserModal } from "@/components/add-user-modal"
import { EditUserModal } from "@/components/edit-user-modal"
import { ViewUserModal } from "@/components/view-user-modal"
import { fetchUsers } from "@/services/userService"

export interface User {
  id: string
  businessName: string
  email: string
  whatsappNumber: string
  address: string
  role: "admin" | "user"
  status: "active" | "inactive" | "suspended"
  createdAt: string
  lastLogin?: string
  totalOrders: number
  totalEarned: string
}

export function UserManagementPage() {
  const [users, setUsers] = useState<User[]>([])
  const [searchTerm, setSearchTerm] = useState("")
  const [statusFilter, setStatusFilter] = useState("all")
  const [roleFilter, setRoleFilter] = useState("all")
  const [currentPage, setCurrentPage] = useState(1)
  const [itemsPerPage, setItemsPerPage] = useState(10)
  const [isAddModalOpen, setIsAddModalOpen] = useState(false)
  const [isEditModalOpen, setIsEditModalOpen] = useState(false)
  const [isViewModalOpen, setIsViewModalOpen] = useState(false)
  const [selectedUser, setSelectedUser] = useState<User | null>(null)

  // Filter users based on search term, status, and role
  const filteredUsers = users.filter((user) => {
    const matchesSearch =
      user.businessName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      user.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
      user.whatsappNumber.includes(searchTerm)

    const matchesStatus = statusFilter === "all" || user.status === statusFilter
    const matchesRole = roleFilter === "all" || user.role === roleFilter

    return matchesSearch && matchesStatus && matchesRole
  })

  // Pagination calculations
  const totalPages = Math.ceil(filteredUsers.length / itemsPerPage)
  const startIndex = (currentPage - 1) * itemsPerPage
  const endIndex = startIndex + itemsPerPage
  const paginatedUsers = filteredUsers.slice(startIndex, endIndex)

  // Reset to first page when filters change
  useEffect(() => {
    setCurrentPage(1)
    loadUsers()
  }, [searchTerm, statusFilter, roleFilter])
  
  const loadUsers = async () => {
    try {
      const usersData = await fetchUsers()
      setUsers(usersData)
    } catch (error) {
      // Error handling is already done in the service
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active":
        return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30"
      case "inactive":
        return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30"
      case "suspended":
        return "bg-red-500/20 text-red-400 border-red-500/30"
      default:
        return "bg-gray-500/20 text-gray-400 border-gray-500/30"
    }
  }

  const getRoleColor = (role: string) => {
    switch (role) {
      case "admin":
        return "bg-purple-500/20 text-purple-400 border-purple-500/30"
      case "user":
        return "bg-blue-500/20 text-blue-400 border-blue-500/30"
      case "seller":
        return "bg-orange-500/20 text-orange-400 border-orange-500/30"
      default:
        return "bg-gray-500/20 text-gray-400 border-gray-500/30"
    }
  }

  const handleViewUser = (user: User) => {
    setSelectedUser(user)
    setIsViewModalOpen(true)
  }

  const handleEditUser = (user: User) => {
    setSelectedUser(user)
    setIsEditModalOpen(true)
  }

  const handleDeleteUser = (userId: string) => {
    const userToDelete = users.find((u) => u.id === userId)
    if (!userToDelete) return

    if (userToDelete.role === "admin") {
      toast.error("Cannot Delete Admin",{
        description: "Admin users cannot be deleted for security reasons",
      })
      return
    }

    setUsers(users.filter((user) => user.id !== userId))
    toast.success("User Deleted",{
      description: `${userToDelete.businessName} has been deleted successfully`,
      className: "bg-emerald-500/10 border-emerald-500/20 text-emerald-400",
    })
  }

  const handleStatusChange = (userId: string, newStatus: "active" | "inactive" | "suspended") => {
    setUsers(users.map((user) => (user.id === userId ? { ...user, status: newStatus } : user)))

    const user = users.find((u) => u.id === userId)
    toast.success("Status Updated",{
      description: `${user?.businessName} status changed to ${newStatus}`,
      className: "bg-violet-500/10 border-violet-500/20 text-violet-400",
    })
  }

  const handleAddUser = async (userData: {
    shop_name: string
    email: string
    password: string
    phone: string
    address: string
    name: string
    whatsapp_number_id: string
    role: string
  }) => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/auth/register`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: userData.name,
          email: userData.email,
          password: userData.password,
          phone: userData.phone,
          address: userData.address,
          shop_name: userData.shop_name,
          whatsapp_number_id: userData.whatsapp_number_id,
          role: userData.role,
        }),
      })

      const data = await response.json()

      if (response.status != 201) {
        throw new Error(data?.message || "Registration failed")
      }
      fetchUsers()

      toast.success("Account created successfully! Redirecting to dashboard...", {
        style: {
          background: "#0f0f23",
          color: "#fff",
        },
      });
      
      setIsAddModalOpen(false)
    } catch (err: any) {
      toast.error("Registration Error",{
        description: err?.message || "Failed to register user",
      })
      throw err
    }
  }

  const handleUpdateUser = async (userData: User) => {
    setUsers(users.map((user) => (user.id === userData.id ? userData : user)))

    setIsEditModalOpen(false)
    setSelectedUser(null)

    toast.success("User Updated",{
      description: `${userData.businessName} has been updated successfully`,
      className: "bg-emerald-500/10 border-emerald-500/20 text-emerald-400",
    })
  }

  const exportUsers = () => {
    // In a real app, this would generate and download a CSV/Excel file
    toast.success("Export Started",{
      description: "User data export has been initiated",
      className: "bg-violet-500/10 border-violet-500/20 text-violet-400",
    })
  }

  // Statistics
  const stats = {
    total: users.length,
    active: users.filter((u) => u.status === "active").length,
    inactive: users.filter((u) => u.status === "inactive").length,
    suspended: users.filter((u) => u.status === "suspended").length,
    admins: users.filter((u) => u.role === "admin").length,
  }

  return (
    <div className="space-y-6 animate-in fade-in-50 duration-500">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold text-white mb-2">User Management</h2>
          <p className="text-gray-400">Manage registered users and their permissions</p>
        </div>
        <div className="flex space-x-3">
          <Button
            onClick={exportUsers}
            variant="outline"
            className="border-gray-600 text-gray-300 hover:text-white hover:border-gray-500 bg-transparent"
          >
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
          <Button
            onClick={() => setIsAddModalOpen(true)}
            className="bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 text-white shadow-lg hover:shadow-violet-500/25 transition-all duration-300"
          >
            <UserPlus className="w-4 h-4 mr-2" />
            Add User
          </Button>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4 animate-in slide-in-from-top duration-500">
        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Total Users</p>
                <p className="text-2xl font-bold text-white">{stats.total}</p>
              </div>
              <Users className="w-8 h-8 text-violet-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Active</p>
                <p className="text-2xl font-bold text-emerald-400">{stats.active}</p>
              </div>
              <CheckCircle className="w-8 h-8 text-emerald-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Inactive</p>
                <p className="text-2xl font-bold text-yellow-400">{stats.inactive}</p>
              </div>
              <div className="w-8 h-8 rounded-full bg-yellow-400/20 flex items-center justify-center">
                <div className="w-4 h-4 rounded-full bg-yellow-400"></div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Suspended</p>
                <p className="text-2xl font-bold text-red-400">{stats.suspended}</p>
              </div>
              <Ban className="w-8 h-8 text-red-400" />
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-sm">Admins</p>
                <p className="text-2xl font-bold text-purple-400">{stats.admins}</p>
              </div>
              <div className="w-8 h-8 rounded-full bg-purple-400/20 flex items-center justify-center">
                <div className="w-4 h-4 rounded-full bg-purple-400"></div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4 animate-in slide-in-from-top duration-500 delay-100">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
          <Input
            placeholder="Search users by name, email, or phone..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 bg-[#1a1a2e] border-gray-600 text-white focus:border-violet-400 focus:ring-violet-400/20 transition-all duration-300"
          />
        </div>

        <Select value={statusFilter} onValueChange={setStatusFilter}>
          <SelectTrigger className="w-40 bg-[#1a1a2e] border-gray-600 text-white">
            <Filter className="w-4 h-4 mr-2" />
            <SelectValue placeholder="Status" />
          </SelectTrigger>
          <SelectContent className="bg-[#1a1a2e] border-gray-600">
            <SelectItem value="all" className="text-white hover:bg-gray-800">
              All Status
            </SelectItem>
            <SelectItem value="active" className="text-white hover:bg-gray-800">
              Active
            </SelectItem>
            <SelectItem value="inactive" className="text-white hover:bg-gray-800">
              Inactive
            </SelectItem>
            <SelectItem value="suspended" className="text-white hover:bg-gray-800">
              Suspended
            </SelectItem>
          </SelectContent>
        </Select>

        <Select value={roleFilter} onValueChange={setRoleFilter}>
          <SelectTrigger className="w-32 bg-[#1a1a2e] border-gray-600 text-white">
            <SelectValue placeholder="Role" />
          </SelectTrigger>
          <SelectContent className="bg-[#1a1a2e] border-gray-600">
            <SelectItem value="all" className="text-white hover:bg-gray-800">
              All Roles
            </SelectItem>
            <SelectItem value="admin" className="text-white hover:bg-gray-800">
              Admin
            </SelectItem>
            <SelectItem value="seller" className="text-white hover:bg-gray-800">
              Seller
            </SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Users Table */}
      <Card className="bg-gradient-to-br from-[#1a1a2e] to-[#16213e] border-gray-700 animate-in slide-in-from-bottom duration-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center justify-between">
            <span>Users ({filteredUsers.length})</span>
            <Select value={itemsPerPage.toString()} onValueChange={(value) => setItemsPerPage(Number(value))}>
              <SelectTrigger className="w-20 h-8 bg-[#0f0f23] border-gray-600 text-white">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-[#1a1a2e] border-gray-600">
                <SelectItem value="5" className="text-white hover:bg-gray-800">
                  5
                </SelectItem>
                <SelectItem value="10" className="text-white hover:bg-gray-800">
                  10
                </SelectItem>
                <SelectItem value="25" className="text-white hover:bg-gray-800">
                  25
                </SelectItem>
                <SelectItem value="50" className="text-white hover:bg-gray-800">
                  50
                </SelectItem>
              </SelectContent>
            </Select>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Business</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Email</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Phone</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Role</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Status</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Orders</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Total Earns</th>
                  <th className="text-left py-3 px-4 text-violet-400 font-semibold">Actions</th>
                </tr>
              </thead>
              <tbody>
                {paginatedUsers.map((user, index) => (
                  <tr
                    key={user.id}
                    className="border-b border-gray-800 hover:bg-gray-800/30 transition-all duration-200 animate-in slide-in-from-left"
                    style={{ animationDelay: `${index * 50}ms` }}
                  >
                    <td className="py-3 px-4">
                      <div>
                        <p className="text-white font-medium">{user.businessName}</p>
                        <p className="text-gray-400 text-sm">Joined {user.createdAt}</p>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-gray-300">{user.email}</td>
                    <td className="py-3 px-4 text-gray-300">{user.whatsappNumber}</td>
                    <td className="py-3 px-4">
                      <Badge className={getRoleColor(user.role)}>{user.role.toUpperCase()}</Badge>
                    </td>
                    <td className="py-3 px-4">
                      <Badge className={getStatusColor(user.status)}>{user.status.toUpperCase()}</Badge>
                    </td>
                    <td className="py-3 px-4 text-emerald-400 font-semibold">{user.totalOrders}</td>
                    <td className="py-3 px-4 text-violet-400 font-semibold">{user.totalEarned}</td>
                    <td className="py-3 px-4">
                      <div className="flex space-x-2">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => handleViewUser(user)}
                          className="text-blue-400 hover:text-blue-300 hover:bg-gray-800/50"
                        >
                          <Eye className="w-4 h-4" />
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => handleEditUser(user)}
                          className="text-violet-400 hover:text-emerald-400 hover:bg-gray-800/50"
                        >
                          <Edit className="w-4 h-4" />
                        </Button>
                        {user.role !== "admin" && (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => handleDeleteUser(user.id)}
                            className="text-red-400 hover:text-red-300 hover:bg-gray-800/50"
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between mt-6 pt-4 border-t border-gray-700">
              <div className="text-sm text-gray-400">
                Showing {startIndex + 1} to {Math.min(endIndex, filteredUsers.length)} of {filteredUsers.length} users
              </div>
              <div className="flex items-center space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(currentPage - 1)}
                  disabled={currentPage === 1}
                  className="border-gray-600 text-gray-300 hover:text-white hover:border-gray-500 bg-transparent"
                >
                  Previous
                </Button>
                <span className="text-gray-400">
                  Page {currentPage} of {totalPages}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage(currentPage + 1)}
                  disabled={currentPage === totalPages}
                  className="border-gray-600 text-gray-300 hover:text-white hover:border-gray-500 bg-transparent"
                >
                  Next
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Modals */}
      <AddUserModal isOpen={isAddModalOpen} onClose={() => setIsAddModalOpen(false)} onAddUser={handleAddUser} />

      <EditUserModal
        isOpen={isEditModalOpen}
        onClose={() => {
          setIsEditModalOpen(false)
          setSelectedUser(null)
        }}
        onUpdateUser={handleUpdateUser}
        user={selectedUser}
      />

      <ViewUserModal
        isOpen={isViewModalOpen}
        onClose={() => {
          setIsViewModalOpen(false)
          setSelectedUser(null)
        }}
        user={selectedUser}
        onStatusChange={handleStatusChange}
      />
    </div>
  )
}
