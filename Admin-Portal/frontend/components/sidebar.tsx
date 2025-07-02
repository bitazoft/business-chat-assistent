"use client"

import { BarChart3, Package, ShoppingCart, Settings, Home, MessageCircle, Menu, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface SidebarProps {
  activeSection: string
  onSectionChange: (section: string) => void
  collapsed: boolean
  onToggleCollapse: () => void
}

const menuItems = [
  { id: "dashboard", label: "Dashboard", icon: Home },
  { id: "products", label: "Products", icon: Package },
  { id: "orders", label: "Orders", icon: ShoppingCart },
  { id: "analytics", label: "Analytics", icon: BarChart3 },
  { id: "settings", label: "Settings", icon: Settings },
]

export function Sidebar({ activeSection, onSectionChange, collapsed, onToggleCollapse }: SidebarProps) {
  return (
    <div
      className={cn(
        "bg-gradient-to-b from-[#1a1a2e] to-[#16213e] border-r border-gray-700 flex flex-col transition-all duration-300 ease-in-out",
        collapsed ? "w-16" : "w-64",
      )}
    >
      {/* Header */}
      <div className="p-4 border-b border-gray-700 flex items-center justify-between">
        {!collapsed && (
          <div className="flex items-center space-x-2">
            <MessageCircle className="w-8 h-8 text-violet-400" />
            <span className="text-xl font-bold text-white">WhatsApp AI</span>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleCollapse}
          className="text-gray-400 hover:text-violet-400 hover:bg-gray-800/50 transition-colors duration-200"
        >
          {collapsed ? <Menu className="w-5 h-5" /> : <X className="w-5 h-5" />}
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {menuItems.map((item) => {
            const Icon = item.icon
            const isActive = activeSection === item.id

            return (
              <li key={item.id}>
                <Button
                  variant="ghost"
                  onClick={() => onSectionChange(item.id)}
                  className={cn(
                    "w-full justify-start transition-all duration-300 group",
                    collapsed ? "px-2" : "px-4",
                    isActive
                      ? "bg-gradient-to-r from-violet-500/20 to-purple-600/20 text-violet-400 border-r-2 border-violet-400"
                      : "text-gray-300 hover:text-violet-400 hover:bg-gray-800/50",
                  )}
                >
                  <Icon className={cn("w-5 h-5 transition-colors duration-200", collapsed ? "mx-auto" : "mr-3")} />
                  {!collapsed && <span className="font-medium transition-all duration-300">{item.label}</span>}
                </Button>
              </li>
            )
          })}
        </ul>
      </nav>

      {/* Collapse indicator */}
      {collapsed && (
        <div className="p-4 border-t border-gray-700">
          <div className="w-8 h-8 mx-auto bg-gradient-to-r from-violet-500 to-purple-600 rounded-full flex items-center justify-center">
            <MessageCircle className="w-4 h-4 text-white" />
          </div>
        </div>
      )}
    </div>
  )
}
