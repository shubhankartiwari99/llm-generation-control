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
  Legend
} from "recharts";

interface EntropyChartProps {
  plainSteps?: TokenStep[];
  adaptiveSteps?: TokenStep[];
}

export default function EntropyChart({ plainSteps, adaptiveSteps }: EntropyChartProps) {
  if ((!plainSteps || plainSteps.length === 0) && (!adaptiveSteps || adaptiveSteps.length === 0)) {
    return (
      <div className="glass-panel panel-content" style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "300px" }}>
        <span className="text-secondary">Run inference to view entropy trace</span>
      </div>
    );
  }

  // Merge the two step arrays by index for the Recharts data format
  const maxLength = Math.max(plainSteps?.length || 0, adaptiveSteps?.length || 0);
  const data = [];
  
  for (let i = 0; i < maxLength; i++) {
    const pStep = plainSteps?.[i];
    const aStep = adaptiveSteps?.[i];
    
    data.push({
      step: i,
      plainEntropy: pStep?.entropy,
      plainToken: pStep?.token?.trim() || (pStep ? "\\n" : ""),
      plainInstability: pStep?.instability,
      adaptiveEntropy: aStep?.entropy,
      adaptiveToken: aStep?.token?.trim() || (aStep ? "\\n" : ""),
      adaptiveInstability: aStep?.instability,
    });
  }

  return (
    <div className="glass-panel panel-content" style={{ height: "400px" }}>
      <h3 style={{ marginBottom: "1rem" }}>Entropy Trace Comparison</h3>
      <div style={{ flex: 1, width: "100%", minHeight: 0 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis dataKey="step" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip
              contentStyle={{ backgroundColor: "#1e1e2d", border: "1px solid #333", borderRadius: "8px" }}
              itemStyle={{ color: "#e2e8f0" }}
              formatter={(value, name, props) => {
                const isPlain = name === "plainEntropy";
                const token = isPlain ? props.payload.plainToken : props.payload.adaptiveToken;
                const label = isPlain ? "Plain" : "Adaptive";
                return [`${Number(value).toFixed(2)} ("${token}")`, label];
              }}
            />
            <Legend wrapperStyle={{ paddingTop: "20px" }} />
            {/* Danger Zone Reference */}
            <ReferenceLine y={1.0} stroke="#ef4444" strokeDasharray="3 3" label={{ position: 'insideTopLeft', value: 'Low Entropy Lock Zone', fill: '#ef4444', fontSize: 12 }} />
            
            {plainSteps && plainSteps.length > 0 && (
              <Line
                type="monotone"
                dataKey="plainEntropy"
                name="plainEntropy"
                stroke="#94a3b8" // Grey for plain
                strokeWidth={2}
                dot={(props: any) => {
                  const { cx, cy, payload } = props;
                  if (payload.plainInstability) return <circle key={`p-${props.index}`} cx={cx} cy={cy} r={6} fill="#ef4444" stroke="none" />;
                  return <circle key={`p-${props.index}`} cx={cx} cy={cy} r={3} fill="#94a3b8" stroke="none" />;
                }}
                activeDot={{ r: 6 }}
              />
            )}

            {adaptiveSteps && adaptiveSteps.length > 0 && (
              <Line
                type="monotone"
                dataKey="adaptiveEntropy"
                name="adaptiveEntropy"
                stroke="#6366f1" // Indigo for adaptive
                strokeWidth={2}
                dot={(props: any) => {
                  const { cx, cy, payload } = props;
                  if (payload.adaptiveInstability) return <circle key={`a-${props.index}`} cx={cx} cy={cy} r={6} fill="#ef4444" stroke="none" />;
                  return <circle key={`a-${props.index}`} cx={cx} cy={cy} r={3} fill="#6366f1" stroke="none" />;
                }}
                activeDot={{ r: 6 }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
