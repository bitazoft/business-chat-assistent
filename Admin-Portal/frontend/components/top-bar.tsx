"use client"

import { Bell, User, LogOut } from "lucide-react"
import { Button } from "@/components/ui/button"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"

interface TopBarProps {
  user: { businessName: string; email: string }
  onLogout: () => void
}

export function TopBar({ user, onLogout }: TopBarProps) {
  return (
    <header className="h-16 bg-gradient-to-r from-[#1a1a2e] to-[#16213e] border-b border-gray-700 flex items-center justify-between px-6">
      <div>
        <h1 className="text-xl font-semibold text-white">{user.businessName}</h1>
        <p className="text-sm text-gray-400">Welcome back!</p>
      </div>

      <div className="flex items-center space-x-4">
        {/* Notifications */}
        <Button
          variant="ghost"
          size="icon"
          className="text-gray-400 hover:text-violet-400 hover:bg-gray-800/50 relative transition-colors duration-200"
        >
          <Bell className="w-5 h-5" />
          <span className="absolute -top-1 -right-1 w-3 h-3 bg-emerald-400 rounded-full animate-pulse"></span>
        </Button>

        {/* Profile Dropdown */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="text-gray-400 hover:text-violet-400 hover:bg-gray-800/50 transition-colors duration-200"
            >
              <User className="w-5 h-5" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="bg-[#1a1a2e] border-gray-700">
            <DropdownMenuItem className="text-gray-300 hover:text-violet-400 hover:bg-gray-800/50 focus:text-violet-400 focus:bg-gray-800/50">
              <User className="w-4 h-4 mr-2" />
              Profile
            </DropdownMenuItem>
            <DropdownMenuItem
              onClick={onLogout}
              className="text-gray-300 hover:text-red-400 hover:bg-gray-800/50 focus:text-red-400 focus:bg-gray-800/50"
            >
              <LogOut className="w-4 h-4 mr-2" />
              Logout
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  )
}
