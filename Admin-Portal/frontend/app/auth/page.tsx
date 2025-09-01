import { Suspense } from "react";
import { AuthPageClient } from "../../components/auth-page-client";

export default function AuthPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <AuthPageClient />
    </Suspense>
  )
}
