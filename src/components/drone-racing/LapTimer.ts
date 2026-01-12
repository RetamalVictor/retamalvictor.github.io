/**
 * Lap Timer
 *
 * Manages lap timing for drone racing, including current lap time,
 * best lap time, and lap completion tracking.
 */

export interface LapTimerState {
    currentLapTime: number;
    bestLapTime: number | null;
    lapCount: number;
}

export class LapTimer {
    private lapStartTime: number = 0;
    private bestLapTime: number = Infinity;
    private lapCount: number = 0;

    /**
     * Start a new lap at the given simulation time
     */
    public startLap(time: number): void {
        this.lapStartTime = time;
    }

    /**
     * Complete the current lap and record the time
     * @param time Current simulation time
     * @returns The completed lap time, or null if invalid
     */
    public completeLap(time: number): number | null {
        const lapTime = time - this.lapStartTime;

        // Only record valid lap times (sanity check)
        if (this.lapStartTime > 0 && lapTime > 1.0) {
            this.lapCount++;
            if (lapTime < this.bestLapTime) {
                this.bestLapTime = lapTime;
            }
            this.lapStartTime = time;
            return lapTime;
        }

        // Start next lap even if this one wasn't valid
        this.lapStartTime = time;
        return null;
    }

    /**
     * Get the best lap time
     * @returns Best lap time in seconds, or null if no laps completed
     */
    public getBestLapTime(): number | null {
        return this.bestLapTime < Infinity ? this.bestLapTime : null;
    }

    /**
     * Get the current lap elapsed time
     * @param currentTime Current simulation time
     */
    public getCurrentLapTime(currentTime: number): number {
        return currentTime - this.lapStartTime;
    }

    /**
     * Get number of completed laps
     */
    public getLapCount(): number {
        return this.lapCount;
    }

    /**
     * Get full timer state for display
     */
    public getState(currentTime: number): LapTimerState {
        return {
            currentLapTime: this.getCurrentLapTime(currentTime),
            bestLapTime: this.getBestLapTime(),
            lapCount: this.lapCount,
        };
    }

    /**
     * Reset the timer (keeps best lap time by default)
     * @param clearBest If true, also clears best lap time
     */
    public reset(clearBest: boolean = false): void {
        this.lapStartTime = 0;
        this.lapCount = 0;

        if (clearBest) {
            this.bestLapTime = Infinity;
        }
    }

    /**
     * Format lap time for display
     */
    public static formatTime(seconds: number | null): string {
        if (seconds === null) {
            return '--';
        }
        return `${seconds.toFixed(1)}s`;
    }
}
