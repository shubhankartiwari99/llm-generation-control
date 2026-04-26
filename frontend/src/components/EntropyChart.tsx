"use client";

import { TokenStep } from "@/types";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

interface EntropyChartProps {
  steps: TokenStep[];
}

export default function EntropyChart({ steps }: EntropyChartProps) {
  if (!steps || steps.length === 0) {
    return (
      <div className="glass-panel panel-content" style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "300px" }}>
        <span className="text-secondary">Run inference to view entropy trace</span>
      </div>
    );
  }

  const data = steps.map((s, i) => ({
    step: i,
    entropy: s.entropy,
    token: s.token.trim() || "\\n",
    instability: s.instability,
  }));

  return (
    <div className="glass-panel panel-content" style={{ height: "350px" }}>
      <h3 style={{ marginBottom: "1rem" }}>Entropy Trace</h3>
      <div style={{ flex: 1, width: "100%", minHeight: 0 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis dataKey="step" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip
              contentStyle={{ backgroundColor: "#1e1e2d", border: "1px solid #333", borderRadius: "8px" }}
              itemStyle={{ color: "#e2e8f0" }}
              labelFormatter={(label, payload) => {
                const item = payload[0]?.payload;
                return `Step ${label}: "${item?.token}"`;
              }}
            />
            {/* Danger Zone Reference */}
            <ReferenceLine y={1.0} stroke="#ef4444" strokeDasharray="3 3" label={{ position: 'insideTopLeft', value: 'Low Entropy Lock Zone', fill: '#ef4444', fontSize: 12 }} />
            
            <Line
              type="monotone"
              dataKey="entropy"
              stroke="#6366f1"
              strokeWidth={2}
              dot={(props: any) => {
                const { cx, cy, payload } = props;
                if (payload.instability) {
                  return (
                    <circle cx={cx} cy={cy} r={6} fill="#ef4444" stroke="none" />
                  );
                }
                return <circle cx={cx} cy={cy} r={3} fill="#6366f1" stroke="none" />;
              }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
