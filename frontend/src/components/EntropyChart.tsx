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

interface ChartPoint {
  step: number;
  plainEntropy?: number;
  plainToken: string;
  plainInstability?: string | null;
  adaptiveEntropy?: number;
  adaptiveToken: string;
  adaptiveInstability?: string | null;
  adaptiveAction?: string | null;
}

interface DotRenderProps {
  cx?: number;
  cy?: number;
  index: number;
  payload: ChartPoint;
}

export default function EntropyChart({ plainSteps, adaptiveSteps }: EntropyChartProps) {
  if ((!plainSteps || plainSteps.length === 0) && (!adaptiveSteps || adaptiveSteps.length === 0)) {
    return (
      <div className="bg-surface-container-lowest border border-border-subtle rounded-xl p-6 flex flex-col items-center justify-center min-h-[300px]">
        <p className="font-body-md text-body-md text-text-secondary opacity-70">Run inference to view entropy trace</p>
      </div>
    );
  }

  // Merge the two step arrays by index for the Recharts data format
  const maxLength = Math.max(plainSteps?.length || 0, adaptiveSteps?.length || 0);
  const data: ChartPoint[] = [];
  
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
      adaptiveAction: aStep?.action,
    });
  }

  return (
    <div className="bg-surface-container-lowest border border-border-subtle rounded-xl p-6 flex flex-col h-[400px]">
      <div className="flex justify-between items-center mb-4">
        <h3 className="font-headline-md text-headline-md text-text-primary m-0">Entropy Trace Comparison</h3>
        <div className="font-label-sm text-label-sm text-text-secondary" aria-label="chart legend">
          🔴 instability · 🟡 low entropy · 🟢 regeneration
        </div>
      </div>
      <div className="flex-1 w-full min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2D2D44" />
            <XAxis dataKey="step" stroke="#94A3B8" tick={{fontFamily: 'JetBrains Mono', fontSize: 11}} />
            <YAxis stroke="#94A3B8" tick={{fontFamily: 'JetBrains Mono', fontSize: 11}} />
            <Tooltip
              contentStyle={{ backgroundColor: "#1A1A2E", border: "1px solid #2D2D44", borderRadius: "8px", fontFamily: 'JetBrains Mono', fontSize: '13px' }}
              itemStyle={{ color: "#FFFFFF" }}
              formatter={(value, name, props) => {
                const isPlain = name === "plainEntropy";
                const token = isPlain ? props.payload.plainToken : props.payload.adaptiveToken;
                const label = isPlain ? "Plain" : "Adaptive";
                return [`${Number(value).toFixed(2)} ("${token}")`, label];
              }}
            />
            <Legend wrapperStyle={{ paddingTop: "20px", fontFamily: 'Hanken Grotesk', fontSize: '14px' }} />
            {/* Danger Zone Reference */}
            <ReferenceLine y={1.0} stroke="#ffb4ab" strokeDasharray="3 3" label={{ position: 'insideTopLeft', value: 'Low Entropy Lock Zone', fill: '#ffb4ab', fontSize: 12, fontFamily: 'JetBrains Mono' }} />
            
            {plainSteps && plainSteps.length > 0 && (
              <Line
                type="monotone"
                dataKey="plainEntropy"
                name="plainEntropy"
                stroke="#94A3B8" // Grey for plain
                strokeWidth={2}
                dot={(props: DotRenderProps) => {
                  const { cx, cy, payload } = props;
                  if (payload.plainInstability) return <circle key={`p-${props.index}`} cx={cx} cy={cy} r={6} fill="#ffb4ab" stroke="none" />;
                  if (payload.plainEntropy !== undefined && payload.plainEntropy < 1) return <circle key={`p-${props.index}`} cx={cx} cy={cy} r={4} fill="#ffb68d" stroke="none" />;
                  return <circle key={`p-${props.index}`} cx={cx} cy={cy} r={3} fill="#94A3B8" stroke="none" />;
                }}
                activeDot={{ r: 6 }}
              />
            )}

            {adaptiveSteps && adaptiveSteps.length > 0 && (
              <Line
                type="monotone"
                dataKey="adaptiveEntropy"
                name="adaptiveEntropy"
                stroke="#c1c1ff" // Primary for adaptive
                strokeWidth={2}
                dot={(props: DotRenderProps) => {
                  const { cx, cy, payload } = props;
                  if (payload.adaptiveInstability) return <circle key={`a-${props.index}`} cx={cx} cy={cy} r={6} fill="#ffb4ab" stroke="none" />;
                  if (payload.adaptiveAction === "regenerate") return <circle key={`a-${props.index}`} cx={cx} cy={cy} r={5} fill="#10B981" stroke="none" />;
                  if (payload.adaptiveEntropy !== undefined && payload.adaptiveEntropy < 1) return <circle key={`a-${props.index}`} cx={cx} cy={cy} r={4} fill="#ffb68d" stroke="none" />;
                  return <circle key={`a-${props.index}`} cx={cx} cy={cy} r={3} fill="#c1c1ff" stroke="none" />;
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
