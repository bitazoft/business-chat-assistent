import React from "react"

interface LoadingSpinnerProps {
  size?: "sm" | "md" | "lg"
  text?: string
  className?: string
}

export function LoadingSpinner({ size = "md", text = "Loading...", className = "" }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: "w-4 h-4",
    md: "w-6 h-6", 
    lg: "w-8 h-8"
  }

  const textSizeClasses = {
    sm: "text-sm",
    md: "text-lg",
    lg: "text-xl"
  }

  return (
    <div className={`min-h-screen bg-gradient-to-br from-[#0f0f23] to-[#1a1a2e] flex items-center justify-center ${className}`}>
      <div className="flex flex-col items-center space-y-4">
        <div className="flex items-center space-x-2">
          <div className={`${sizeClasses[size]} bg-violet-400 rounded-full animate-pulse`}></div>
          <div className={`${sizeClasses[size]} bg-emerald-400 rounded-full animate-pulse delay-100`}></div>
          <div className={`${sizeClasses[size]} bg-violet-400 rounded-full animate-pulse delay-200`}></div>
        </div>
        <div className={`text-violet-400 ${textSizeClasses[size]} font-medium animate-pulse`}>{text}</div>
      </div>
    </div>
  )
}