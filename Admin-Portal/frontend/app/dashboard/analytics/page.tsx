"use client"

import { AnalyticsPage } from "@/components/analytics-page"
import { RevenueAnalyticsPage } from "@/components/revenue-analytics-page"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function AnalyticsPageRoute() {
  return (
    <div className="space-y-6">
      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-2 bg-[#1a1a2e] border-gray-600">
          <TabsTrigger 
            value="overview" 
            className="text-white data-[state=active]:bg-violet-600 data-[state=active]:text-white"
          >
            Overview Analytics
          </TabsTrigger>
          <TabsTrigger 
            value="revenue" 
            className="text-white data-[state=active]:bg-violet-600 data-[state=active]:text-white"
          >
            Revenue & Orders
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview" className="mt-6">
          <AnalyticsPage />
        </TabsContent>
        
        <TabsContent value="revenue" className="mt-6">
          <RevenueAnalyticsPage />
        </TabsContent>
      </Tabs>
    </div>
  )
}
