import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Activity } from "lucide-react";

export default function LoginView({ email, setEmail, password, setPassword, doLogin, authErr }: { email: string; setEmail: (v: string) => void; password: string; setPassword: (v: string) => void; doLogin: () => void; authErr: string | null; }) {
  return (
    <div className="max-w-md w-full mx-auto px-4">
      <Card className="backdrop-blur-lg bg-white/90 border-blue-100 shadow-xl">
        <CardHeader className="text-center pb-4">
          <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center mx-auto mb-4">
            <Activity className="w-8 h-8 text-white" />
          </div>
          <CardTitle className="text-2xl text-gray-800">Welcome Back</CardTitle>
          <p className="text-gray-600">Sign in to continue to LBC Analysis</p>
        </CardHeader>
        <CardContent className="space-y-4">
          <Input type="email" placeholder="Email address" value={email} onChange={(e) => setEmail(e.target.value)} className="h-12" />
          <Input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} className="h-12" />
          <Button onClick={doLogin} className="w-full h-12 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700">Sign In</Button>
          {authErr && <p className="text-sm text-red-600 text-center">{authErr}</p>}
        </CardContent>
      </Card>
    </div>
  );
}