/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI, Type } from '@google/genai';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
app.use(express.json());

const PORT = 3000;

// Lazy initialize Gemini Client to prevent crash if key is missing on startup
let aiClient: GoogleGenAI | null = null;

function getGeminiClient(): GoogleGenAI {
  if (!aiClient) {
    const key = process.env.GEMINI_API_KEY;
    if (!key || key === 'MY_GEMINI_API_KEY' || key.trim() === '') {
      throw new Error('GEMINI_API_KEY environment variable is not configured. Please add it in the Secrets panel.');
    }
    aiClient = new GoogleGenAI({
      apiKey: key,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }
  return aiClient;
}

// Current system time context for Gemini
const CURRENT_TIME_CONTEXT = "2026-06-30T05:34:25-07:00 (Tuesday)";

/**
 * 1. AI Priority and Risk Engine Endpoint
 * Evaluates urgency, effort, and deadline proximity to calculate risk and priorities.
 */
app.post('/api/ai/prioritize', async (req, res) => {
  try {
    const { tasks } = req.body;
    if (!tasks || !Array.isArray(tasks) || tasks.length === 0) {
      return res.json({ tasks: [] });
    }

    try {
      const ai = getGeminiClient();
      const response = await ai.models.generateContent({
        model: 'gemini-3.5-flash',
        contents: `Analyze the following tasks based on current date/time context: ${CURRENT_TIME_CONTEXT}.
        Calculate a deadline risk score (0-100) where 100 is extremely high risk (overdue, or impossible to finish in time given the effort).
        Provide a concise, specific explanation (max 12 words) of why this risk is high/medium/low, taking into account:
        - Remaining time till deadline vs Estimated effort (hours).
        - Priority and status.
        - Categorical risk factor.
        
        Tasks list:
        ${JSON.stringify(tasks.map(t => ({ id: t.id, title: t.title, description: t.description, deadline: t.deadline, priority: t.priority, effort: t.effort, category: t.category, status: t.status })))}`,
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              analyses: {
                type: Type.ARRAY,
                description: 'Risk and priority analysis for each task.',
                items: {
                  type: Type.OBJECT,
                  properties: {
                    id: { type: Type.STRING },
                    riskScore: { type: Type.INTEGER, description: 'Score from 0 to 100.' },
                    riskExplanation: { type: Type.STRING, description: 'Explanation under 12 words, e.g., "Only 3 hours left for a 6-hour task."' },
                    priorityLevel: { type: Type.STRING, description: 'Recalculated priority: low, medium, high, or critical.' },
                  },
                  required: ['id', 'riskScore', 'riskExplanation', 'priorityLevel'],
                },
              },
            },
            required: ['analyses'],
          },
        },
      });

      const parsed = JSON.parse(response.text || '{}');
      return res.json(parsed);
    } catch (apiError: any) {
      console.warn('Gemini Prioritize API Error (using fallback):', apiError.message);
      
      // Fallback local priority calculation if API key is unconfigured or fails
      const fallbackAnalyses = tasks.map(t => {
        if (t.status === 'completed') {
          return { id: t.id, riskScore: 0, riskExplanation: 'Task completed successfully', priorityLevel: t.priority };
        }
        
        const now = new Date("2026-06-30T05:34:25-07:00");
        const deadline = new Date(t.deadline);
        const hoursLeft = Math.max(0.1, (deadline.getTime() - now.getTime()) / (1000 * 60 * 60));
        const ratio = t.effort / hoursLeft;
        
        let riskScore = Math.min(100, Math.round(ratio * 50));
        if (hoursLeft <= 0) riskScore = 100;
        
        let riskExplanation = 'On track to finish.';
        if (riskScore > 80) riskExplanation = `High risk! Only ${Math.round(hoursLeft)} hrs left for ${t.effort} hrs of work.`;
        else if (riskScore > 45) riskExplanation = `Moderate risk. Schedule is tightening.`;
        
        let priorityLevel = t.priority;
        if (riskScore > 75) priorityLevel = 'critical';
        else if (riskScore > 40) priorityLevel = 'high';

        return { id: t.id, riskScore, riskExplanation, priorityLevel };
      });

      return res.json({ analyses: fallbackAnalyses, isFallback: true, error: apiError.message });
    }
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * 2. Rescue Planner Endpoint
 * Generates an actionable step-by-step roadmap and schedules them into daily time blocks.
 */
app.post('/api/ai/rescue', async (req, res) => {
  try {
    const { task } = req.body;
    if (!task) {
      return res.status(400).json({ error: 'Task data is required' });
    }

    try {
      const ai = getGeminiClient();
      const response = await ai.models.generateContent({
        model: 'gemini-3.5-flash',
        contents: `Create an emergency rescue plan for this high-risk task. Break it down into 3-5 bite-sized logical subtasks (with estimated hours) and schedule concrete, distraction-free focus time blocks (starting from today).
        
        Current context time: ${CURRENT_TIME_CONTEXT}
        Task details:
        Title: ${task.title}
        Description: ${task.description}
        Category: ${task.category}
        Priority: ${task.priority}
        Estimated Effort: ${task.effort} hours
        Deadline: ${task.deadline}
        
        The result should include:
        1. A brief proactive coaching summary (max 2 sentences) encouraging the user and explaining the strategy.
        2. Subtasks (detailed list of milestones).
        3. Time blocks: scheduling specific focus sessions starting from today. Provide "daysFromNow" (0 for today, 1 for tomorrow, etc.) and recommended start times (e.g. "09:00", "14:30").`,
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              summary: { type: Type.STRING, description: 'Rescue strategy coaching summary.' },
              subtasks: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    title: { type: Type.STRING },
                    estimatedHours: { type: Type.NUMBER },
                  },
                  required: ['title', 'estimatedHours'],
                },
              },
              timeBlocks: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    title: { type: Type.STRING, description: 'Focus block theme, e.g., "Deep Work: Project Drafting"' },
                    durationHours: { type: Type.NUMBER },
                    daysFromNow: { type: Type.INTEGER, description: '0 for today, 1 for tomorrow, etc.' },
                    startTime: { type: Type.STRING, description: 'Format HH:MM' },
                  },
                  required: ['title', 'durationHours', 'daysFromNow', 'startTime'],
                },
              },
            },
            required: ['summary', 'subtasks', 'timeBlocks'],
          },
        },
      });

      const parsed = JSON.parse(response.text || '{}');
      return res.json(parsed);
    } catch (apiError: any) {
      console.warn('Gemini Rescue API Error (using fallback):', apiError.message);
      
      // Fallback local rescue plan generator
      const halfEffort = Math.max(1, Math.round(task.effort / 2));
      const remainingEffort = Math.max(1, task.effort - halfEffort);
      
      const fallbackRescue = {
        summary: `AI Rescue Mode: We've split your "${task.title}" task into milestones with structured focus blocks to protect your deadline. Let's tackle this step-by-step!`,
        subtasks: [
          { title: 'Phase 1: Initial research & core drafting', estimatedHours: halfEffort },
          { title: 'Phase 2: Finalizing deliverables & revision', estimatedHours: remainingEffort },
          { title: 'Phase 3: Ultimate review and submission readiness check', estimatedHours: 0.5 }
        ],
        timeBlocks: [
          { title: `Focus: ${task.title} Setup & Phase 1`, durationHours: halfEffort, daysFromNow: 0, startTime: '10:00' },
          { title: `Focus: ${task.title} Phase 2 & Review`, durationHours: remainingEffort + 0.5, daysFromNow: 1, startTime: '14:00' }
        ],
        isFallback: true,
        error: apiError.message
      };

      return res.json(fallbackRescue);
    }
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * 3. Voice / Natural Language Productivity Input
 * Parses conversational inputs into fully structured tasks.
 */
app.post('/api/ai/voice-input', async (req, res) => {
  try {
    const { text } = req.body;
    if (!text || text.trim() === '') {
      return res.status(400).json({ error: 'Text input is required' });
    }

    try {
      const ai = getGeminiClient();
      const response = await ai.models.generateContent({
        model: 'gemini-3.5-flash',
        contents: `Analyze this conversational productivity input and extract a fully structured task.
        
        Current context time is: ${CURRENT_TIME_CONTEXT}
        Input text: "${text}"
        
        Map the fields logically:
        - Title: concise, action-oriented.
        - Description: brief context of what's needed.
        - Deadline: calculate the exact ISO Date (YYYY-MM-DD) based on natural statements like "next Friday", "tomorrow night", "by 5 PM".
        - Priority: low, medium, high, or critical.
        - Effort: reasonable estimated hours to complete (integer or decimal).
        - Category: choose one of: study, work, personal, finance, health, other.`,
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              title: { type: Type.STRING },
              description: { type: Type.STRING },
              deadline: { type: Type.STRING, description: 'ISO date string (YYYY-MM-DD).' },
              priority: { type: Type.STRING, description: 'low, medium, high, or critical' },
              effort: { type: Type.NUMBER, description: 'Estimated hours to finish' },
              category: { type: Type.STRING, description: 'study, work, personal, finance, health, other' },
            },
            required: ['title', 'description', 'deadline', 'priority', 'effort', 'category'],
          },
        },
      });

      const parsed = JSON.parse(response.text || '{}');
      return res.json(parsed);
    } catch (apiError: any) {
      console.warn('Gemini Voice Input API Error (using fallback):', apiError.message);
      
      // Fallback simple keyword matching for natural language parsing
      const cleanText = text.toLowerCase();
      let category = 'personal';
      if (cleanText.includes('exam') || cleanText.includes('study') || cleanText.includes('assignment') || cleanText.includes('class')) {
        category = 'study';
      } else if (cleanText.includes('work') || cleanText.includes('meeting') || cleanText.includes('project') || cleanText.includes('report')) {
        category = 'work';
      } else if (cleanText.includes('bill') || cleanText.includes('pay') || cleanText.includes('rent') || cleanText.includes('finance')) {
        category = 'finance';
      }

      let priority = 'medium';
      if (cleanText.includes('urgent') || cleanText.includes('asap') || cleanText.includes('critical') || cleanText.includes('tonight')) {
        priority = 'high';
      }

      let effort = 2;
      const hoursMatch = cleanText.match(/(\d+)\s*hour/);
      if (hoursMatch) {
        effort = parseInt(hoursMatch[1]);
      }

      const today = new Date("2026-06-30");
      let deadlineDate = new Date(today);
      if (cleanText.includes('tomorrow')) {
        deadlineDate.setDate(today.getDate() + 1);
      } else if (cleanText.includes('next week') || cleanText.includes('friday')) {
        deadlineDate.setDate(today.getDate() + 3);
      } else {
        deadlineDate.setDate(today.getDate() + 2); // Default 2 days from now
      }

      const fallbackTask = {
        title: text.substring(0, 40) + (text.length > 40 ? '...' : ''),
        description: `Extracted from voice prompt: "${text}"`,
        deadline: deadlineDate.toISOString().split('T')[0],
        priority,
        effort,
        category,
        isFallback: true,
        error: apiError.message
      };

      return res.json(fallbackTask);
    }
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

/**
 * 4. Context-Aware Reminders and Proactive Suggestions Endpoint
 * Automatically identifies slip risk, daily planning recommendations, or habit check-ins.
 */
app.post('/api/ai/suggestions', async (req, res) => {
  try {
    const { tasks, goals } = req.body;
    
    try {
      const ai = getGeminiClient();
      const response = await ai.models.generateContent({
        model: 'gemini-3.5-flash',
        contents: `Analyze the user's workload and goals. Provide 3-4 proactive, highly motivating suggestions or urgent warnings.
        
        Current date/time context: ${CURRENT_TIME_CONTEXT}
        Active Tasks:
        ${JSON.stringify((tasks || []).map(t => ({ title: t.title, deadline: t.deadline, status: t.status, riskScore: t.riskScore, priority: t.priority })))}
        
        Active Goals:
        ${JSON.stringify((goals || []).map(g => ({ title: g.title, progress: g.progress, targetDate: g.targetDate })))}
        
        Generate exactly 3 or 4 dynamic suggestions. Ensure:
        - At least one "risk" warning if any task has high riskScore.
        - One "reorganize" suggestion if multiple tasks are due around the same time.
        - One "encouragement" or "reminder" to maintain streaks or work toward goals.
        - "actionText" must be short and direct (e.g., "Start Rescue Plan", "Snooze non-essentials", "Log Progress").
        - If applicable, include "taskId" matching an active task's ID so the user can quickly act.`,
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              suggestions: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    id: { type: Type.STRING },
                    type: { type: Type.STRING, description: 'risk, reorganize, reminder, or encouragement' },
                    message: { type: Type.STRING, description: 'Actionable suggestion message under 16 words.' },
                    actionText: { type: Type.STRING, description: 'Short CTA button text (2-3 words).' },
                    taskId: { type: Type.STRING, description: 'Optional ID of the related task.' },
                  },
                  required: ['id', 'type', 'message', 'actionText'],
                },
              },
            },
            required: ['suggestions'],
          },
        },
      });

      const parsed = JSON.parse(response.text || '{}');
      return res.json(parsed);
    } catch (apiError: any) {
      console.warn('Gemini Suggestions API Error (using fallback):', apiError.message);
      
      const fallbackSuggestions = [
        {
          id: 's1',
          type: 'risk',
          message: 'Your Physics Exam Prep is under high deadline risk! Only 1 day left.',
          actionText: 'Activate Rescue Plan',
          taskId: 't1'
        },
        {
          id: 's2',
          type: 'reorganize',
          message: 'Workload looks heavy this afternoon. Consider rescheduling non-essential admin tasks.',
          actionText: 'Reorganize Day'
        },
        {
          id: 's3',
          type: 'encouragement',
          message: 'Awesome job! You are on a 5-day streak for "Daily Gym Study". Keep it up!',
          actionText: 'Log Habits'
        }
      ];

      return res.json({ suggestions: fallbackSuggestions, isFallback: true, error: apiError.message });
    }
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// Serve frontend build static files in production or hook up Vite dev middleware
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Finishline AI server running at http://0.0.0.0:${PORT} under ${process.env.NODE_ENV || 'development'} mode.`);
  });
}

startServer();
