// Copyright (c) 2024 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Thomas Hahn, Alexander Hampel, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides a C++ wrapper class for an `MPI_Comm` object.
 */

#pragma once

#include "./communicator.hpp"

#include <mpi.h>

#include <cassert>

#include <cstring>

namespace mpi {

  template <class BaseType> class shared_window;

  /// The window class
  template <class BaseType>
  class window {
    friend class shared_window<BaseType>;
    MPI_Win win{MPI_WIN_NULL};
    BaseType* base_local{nullptr};
  public:
    window() = default;
    window(window const&) = delete;
    window(window &&other) noexcept : win{std::exchange(other.win, MPI_WIN_NULL)} {}
    window& operator=(window const&) = delete;
    window& operator=(window &&rhs) noexcept {
      if (this != std::addressof(rhs)) {
        this->free();
        this->win = std::exchange(rhs.win, MPI_WIN_NULL);
      }
      return *this;
    }

    /// Create a window over an existing local memory buffer
    explicit window(communicator &c, BaseType *base, MPI_Aint size = 0) noexcept {
      if (has_env) {
        MPI_Win_create(base, size * sizeof(BaseType), alignof(BaseType), MPI_INFO_NULL, c.get(), &win);
      } else {
        base_local = (BaseType*) std::aligned_alloc(alignof(BaseType), size * sizeof(BaseType));
        if (base_local == nullptr) {
          std::abort(); // failed memory allocation
        }
      }
    }

    /// Create a window and allocate memory for a local memory buffer
    explicit window(communicator &c, MPI_Aint size = 0) noexcept {
      if (has_env) {
        void *baseptr = nullptr;
        MPI_Win_allocate(size * sizeof(BaseType), alignof(BaseType), MPI_INFO_NULL, c.get(), &baseptr, &win);
      } else {
        //needs to be changed
        base_local = (BaseType*) std::aligned_alloc(alignof(BaseType), size * sizeof(BaseType));
        if (base_local == nullptr) {
          std::abort();// failed memory allocation
        }
      }
    }

    ~window() { free(); }

    explicit operator MPI_Win() const noexcept { return win; };
    explicit operator MPI_Win*() noexcept { return &win; };

    void free() noexcept {
      if (win != MPI_WIN_NULL) {
        MPI_Win_free(&win);
      } else {
        std::free(base_local);
      }
    }

    /// Synchronization routine in active target RMA. It opens and closes an access epoch.
    void fence(int assert = 0) const noexcept {
      if (has_env) {
        MPI_Win_fence(assert, win);
      }
    }

    /// Complete all outstanding RMA operations at both the origin and the target
    void flush(int rank = -1) const noexcept {
      if (has_env) {
        if (rank < 0) {
          MPI_Win_flush_all(win);
        } else {
          MPI_Win_flush(rank, win);
        } 
      }
    }

    /// Synchronize the private and public copies of the window
    void sync() const noexcept {
        if (has_env) {      
          MPI_Win_sync(win);
        }
    }

    /// Starts an RMA access epoch locking access to a particular or all ranks in the window
    void lock(int rank = -1, int lock_type = MPI_LOCK_SHARED, int assert = 0) const noexcept {
      if (has_env) {
        if (rank < 0) {
          MPI_Win_lock_all(assert, win);
        } else {
          MPI_Win_lock(lock_type, rank, assert, win);
        }
      }
    }

    /// Completes an RMA access epoch started by a call to lock()
    void unlock(int rank = -1) const noexcept {
      if (has_env) {
        if (rank < 0) {
        MPI_Win_unlock_all(win);
        } else {
          MPI_Win_unlock(rank, win);
        }
      }
    }

    /// Load data from a remote memory window.
    template <typename TargetType = BaseType, typename OriginType>
    std::enable_if_t<has_mpi_type<OriginType> && has_mpi_type<TargetType>, void>
    get(OriginType *origin_addr, int origin_count, int target_rank, MPI_Aint target_disp = 0, int target_count = -1) const noexcept {
      if (has_env) {
        MPI_Datatype origin_datatype = mpi_type<OriginType>::get();
        MPI_Datatype target_datatype = mpi_type<TargetType>::get();
        int target_count_ = target_count < 0 ? origin_count : target_count;
        MPI_Get(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count_, target_datatype, win);
      } else {
        if (sizeof(OriginType) != sizeof(BaseType)) {
          std::abort();
        }
        std::memcpy(origin_addr, base_local, sizeof(OriginType) * origin_count);
      }
    }

    /// Store data to a remote memory window.
    template <typename TargetType = BaseType, typename OriginType>
    std::enable_if_t<has_mpi_type<OriginType> && has_mpi_type<TargetType>, void>
    put(OriginType *origin_addr, int origin_count, int target_rank, MPI_Aint target_disp = 0, int target_count = -1) const noexcept {
      if (has_env) {
        MPI_Datatype origin_datatype = mpi_type<OriginType>::get();
        MPI_Datatype target_datatype = mpi_type<TargetType>::get();
        int target_count_ = target_count < 0 ? origin_count : target_count;
        MPI_Put(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count_, target_datatype, win);
      } else {
        if (sizeof(OriginType) != sizeof(BaseType)) {
          std::abort(); 
        }
        std::memcpy(base_local + target_disp, origin_addr, sizeof(OriginType) * origin_count);
      }
    }

    /// Accumulate data into target process through remote memory access.
    template <typename TargetType = BaseType, typename OriginType>
    std::enable_if_t<has_mpi_type<OriginType> && has_mpi_type<TargetType>, void>
    accumulate(OriginType const *origin_addr, int origin_count, int target_rank, MPI_Aint target_disp = 0, int target_count = -1, MPI_Op op = MPI_SUM) const noexcept {
      if (has_env) {
        MPI_Datatype origin_datatype = mpi_type<OriginType>::get();
        MPI_Datatype target_datatype = mpi_type<TargetType>::get();
        int target_count_ = target_count < 0 ? origin_count : target_count;
        MPI_Accumulate(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count_, target_datatype, op, win);
      } else {
        if (sizeof(OriginType) != sizeof(BaseType)) {
          std::abort(); 
        } 

        switch (op) {
          case MPI_SUM:
            for (int i = 0; i < origin_count; ++i) {
              base_local[target_disp + i] += origin_addr[i];
            }
            break;
          case MPI_PROD:
            for (int i = 0; i < origin_count; ++i) {
               base_local[target_disp + i] *= origin_addr[i];   
            }
            break;
          case MPI_MIN:
            for (int i = 0; i < origin_count; ++i) {
              base_local[target_disp + i] = std::min(base_local[target_disp + i], origin_addr[i]);
            }
            break;
          case MPI_MAX:
            for (int i = 0; i < origin_count; ++i) {
              base_local[target_disp + i] = std::max(base_local[target_disp + i], origin_addr[i]);
            }
            break;
          default:
              std::abort(); 
        }
      }
    }

    /// Obtains the value of a window attribute.
    void* get_attr(int win_keyval) const noexcept {
      if (has_env) {
        int flag;
        void *attribute_val;
        MPI_Win_get_attr(win, win_keyval, &attribute_val, &flag);
        assert(flag);
        return attribute_val;
      } else {
        return nullptr;
      }
    }

    /// Set the value of a window attribute.
    void set_attr(int win_keyval, void* attribute_val) {
      if (has_env) {
        MPI_Win_set_attr(win, win_keyval, attribute_val);
      }
    }

    // Expose some commonly used attributes
    BaseType* base() const noexcept { return static_cast<BaseType*>(get_attr(MPI_WIN_BASE)); }
    MPI_Aint size() const noexcept { return *static_cast<MPI_Aint*>(get_attr(MPI_WIN_SIZE)); }
    int disp_unit() const noexcept { return *static_cast<int*>(get_attr(MPI_WIN_DISP_UNIT)); }
  };

  /// The shared_window class
  template <class BaseType>
  class shared_window : public window<BaseType> {
  public:
    shared_window() = default;

    /// Create a window and allocate memory for a shared memory buffer
    explicit shared_window(shared_communicator& c, MPI_Aint size) noexcept {
      if (has_env) {
        void* baseptr = nullptr;
        MPI_Win_allocate_shared(size * sizeof(BaseType), alignof(BaseType), MPI_INFO_NULL, c.get(), &baseptr, &(this->win));
      } else {
        this->base_local = (BaseType*) std::malloc(size * sizeof(BaseType));
        if (this->base_local == nullptr) {
          std::abort();
        }
      }
    }

    /// Query a shared memory window
    std::tuple<MPI_Aint, int, void*> query(int rank = MPI_PROC_NULL) const noexcept {
      if (has_env) {
        MPI_Aint size = 0;
        int disp_unit = 0;
        void *baseptr = nullptr;
        MPI_Win_shared_query(this->win, rank, &size, &disp_unit, &baseptr);
        return {size, disp_unit, baseptr};
      } else {
        return {sizeof(BaseType), sizeof(BaseType), this->base_local};
      }
    }

    // Override the commonly used attributes of the window base class
    BaseType* base(int rank = MPI_PROC_NULL) const noexcept { return static_cast<BaseType*>(std::get<2>(query(rank))); }
    MPI_Aint size(int rank = MPI_PROC_NULL) const noexcept { return std::get<0>(query(rank)) / sizeof(BaseType); }
    int disp_unit(int rank = MPI_PROC_NULL) const noexcept { return std::get<1>(query(rank)); }
  };

};
