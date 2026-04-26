//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// This example is provided by the Geant4-DNA collaboration
// Any report or published results obtained using the Geant4-DNA software
// shall cite the following Geant4-DNA collaboration publications:
// Med. Phys. 45 (2018) e722-e739
// Phys. Med. 31 (2015) 861-874
// Med. Phys. 37 (2010) 4692-4708
// Int. J. Model. Simul. Sci. Comput. 1 (2010) 157–178
//
// The Geant4-DNA web site is available at http://geant4-dna.org
//
/// \file SteppingAction.hh
/// \brief Definition of the SteppingAction class

#ifndef SteppingAction_h
#define SteppingAction_h 1

#include "G4UserSteppingAction.hh"
#include <set>
#include <string>
#include <vector>

class EventAction;
class G4Track;

class SteppingAction : public G4UserSteppingAction
{
  public:
    explicit SteppingAction(EventAction* eventAction);
    virtual ~SteppingAction();

    virtual void UserSteppingAction(const G4Step*);

    struct StepRecord {
      int stepNo;
      double kinE_eV;
      std::string process;
      std::string channel; // optional sub-channel/model if derivable
      std::string model;
      double sigma_area_cm2; // microscopic cross section (cm^2), -1 if not available
    };

    // Accessors for collected per-step logs (used by RunAction to print after run)
    static void SetLoggingEnabled(bool enabled);
    static bool IsLoggingEnabled();
    static std::vector<StepRecord>& Logs();
    static void ClearLogs();

    static void ClearObservedModels();
    static std::vector<std::string> ObservedModels();

    /// Reset cached fallback-activation state so the next call to
    /// SetHighEnergyFallbackActive will unconditionally re-evaluate.
    static void ResetHighEnergyFallbackState();
    /// Enable / disable standard-EM fallback processes (eIoni, eBrem, msc,
    /// CoulombScat) based on the track's current kinetic energy vs 10 MeV.
    static void SetHighEnergyFallbackActive(const G4Track* track);

  private:
    EventAction* fEventAction;
};
#endif
