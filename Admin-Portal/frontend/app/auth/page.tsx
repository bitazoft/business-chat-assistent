import { Suspense } from "react";
import { AuthPageClient } from "../../components/auth-page-client";
import { LoadingSpinner } from "@/components/ui/loading-spinner";

export default function AuthPage() {
  return (
    <Suspense fallback={<LoadingSpinner size="md" />}>
      <AuthPageClient />
    </Suspense>
  )
}
